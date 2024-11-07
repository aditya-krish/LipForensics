import logging
import os
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import av
import cv2
import face_alignment
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from skimage import transform as tf
from torchvision import transforms
from torchvision.transforms import CenterCrop, Compose


# ============= Configuration =============
@dataclass
class ProcessingConfig:
    """Configuration for video processing parameters"""

    window_margin: int = 12
    crop_height: int = 88
    crop_width: int = 88
    frames_per_clip: int = 25
    face_margin_factor: float = 0.25
    start_idx: int = 48  # Start of mouth landmarks
    stop_idx: int = 68  # End of mouth landmarks
    stable_points: List[int] = field(default_factory=lambda: [33, 36, 39, 42, 45])
    std_size: Tuple[int, int] = (256, 256)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    debug_mode: bool = False


# ============= Video Transforms =============
class ToTensorVideo:
    def __call__(self, clip):
        return clip.float().permute(3, 0, 1, 2) / 255.0


class NormalizeVideo:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        clip = clip.clone()
        mean = torch.as_tensor(self.mean, dtype=clip.dtype, device=clip.device)
        std = torch.as_tensor(self.std, dtype=clip.dtype, device=clip.device)
        clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return clip


# ============= Main Processor Class =============
class VideoProcessor:
    """Handles the processing pipeline for deepfake detection"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.device = torch.device(config.device)

        # Initialize models
        self.logger.info("Initializing models...")
        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            select_largest=True,
            post_process=False,
        )
        self.landmark_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=self.config.device,
        )
        self.ort_session = None

        # Setup transforms
        self.transform = Compose(
            [ToTensorVideo(), CenterCrop((88, 88)), NormalizeVideo((0.421,), (0.165,))]
        )

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG if self.config.debug_mode else logging.INFO)
        return logger

    def load_model(self, onnx_path: str):
        """Load the ONNX model"""
        self.logger.info(f"Loading ONNX model from {onnx_path}")
        try:
            # Create ONNX Runtime session
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {str(e)}")
            raise

    def _load_video_frames(self, video_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess video frames using a generator to save memory"""
        self.logger.debug("Loading video frames")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame

        finally:
            cap.release()

    def process_video(self, video_path: Union[str, Path]) -> Dict[int, float]:
        """Process video and return forgery probabilities for each face"""
        self.logger.info(f"Processing video: {video_path}")

        normalize_clip = NormalizeVideo((0.421,), (0.165,))

        try:
            # Initialize video capture using av
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            # Set stream parameters for faster decoding
            stream.thread_type = "AUTO"  # Enable multithreaded decoding
            container.streams.video[0].thread_count = max(1, os.cpu_count() // 2)

            # Initialize batch processing
            current_batch = []
            batch_size = 110  # Keep original batch size
            frame_count = 0
            all_face_frames = {}  # Accumulate frames for each face across batches

            # Decode frames in batches
            for frame in container.decode(video=0):
                frame_count += 1
                frame_idx = frame_count - 1

                # Convert to RGB numpy array efficiently
                frame_rgb = frame.to_ndarray(format="rgb24")
                current_batch.append(frame_rgb)

                # Process when batch is full
                if len(current_batch) == batch_size:
                    self.logger.debug(
                        f"Processing batch of {len(current_batch)} frames"
                    )

                    # Process batch of frames using numpy array
                    try:
                        batch_face_frames = self._process_mouth_regions_frames(
                            np.array(current_batch)
                        )
                        # Merge frames from this batch into all_face_frames
                        for face_id, frames in batch_face_frames.items():
                            if face_id not in all_face_frames:
                                all_face_frames[face_id] = []
                            all_face_frames[face_id].extend(frames)
                    except Exception as e:
                        self.logger.error(f"Error processing batch: {str(e)}")

                    # Clear batch
                    current_batch = []

                if frame_idx >= 120:
                    break

            # Process any remaining frames in the last batch
            if current_batch:
                try:
                    batch_face_frames = self._process_mouth_regions_frames(
                        np.array(current_batch)
                    )
                    for face_id, frames in batch_face_frames.items():
                        if face_id not in all_face_frames:
                            all_face_frames[face_id] = []
                        all_face_frames[face_id].extend(frames)
                except Exception as e:
                    self.logger.error(f"Error processing final batch: {str(e)}")

            # Process final probabilities for all accumulated frames
            face_probabilities = {}
            for face_id, frames in all_face_frames.items():
                self.logger.info(f"\nProcessing face ID {face_id}:")
                self.logger.info(f"Total frames collected for face: {len(frames)}")

                if len(frames) >= self.config.frames_per_clip:
                    clips, lengths = self._create_clips(frames)
                    self.logger.info(
                        f"Created {len(clips)} clips from {len(frames)} frames"
                    )

                    face_probs = []
                    for clip_idx, (clip, length) in enumerate(zip(clips, lengths)):
                        if clip is not None and clip.size(0) > 0:
                            clip = normalize_clip(clip)
                            prob = self._run_inference([clip], [length])
                            if prob is not None and not math.isnan(prob):
                                face_probs.append(prob)
                                self.logger.info(
                                    f"Clip {clip_idx}: probability = {prob:.4f}"
                                )
                            else:
                                self.logger.warning(
                                    f"Clip {clip_idx}: Invalid probability detected"
                                )
                        else:
                            self.logger.warning(
                                f"Clip {clip_idx}: Invalid clip detected"
                            )

                    if face_probs:
                        final_prob = np.mean(face_probs)
                        face_probabilities[face_id] = final_prob
                        self.logger.info(
                            f"Face {face_id} final probability (mean of {len(face_probs)} clips): {final_prob:.4f}"
                        )
                    else:
                        self.logger.warning(
                            f"No valid probabilities collected for face {face_id}"
                        )
                else:
                    self.logger.warning(
                        f"Insufficient frames for face {face_id}. "
                        f"Got {len(frames)}, need at least {self.config.frames_per_clip}"
                    )

            self.logger.debug(f"Processed {frame_count} frames total")
            self.logger.debug(f"Face probabilities collected: {face_probabilities}")

            if not face_probabilities:
                raise ValueError("No valid predictions were made")

            self.logger.info(
                f"Inference complete. Probabilities per face: {face_probabilities}"
            )
            return face_probabilities

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise

    def _process_mouth_regions_frames(
        self, frames: torch.Tensor
    ) -> Dict[int, List[torch.Tensor]]:
        """Extract mouth regions from frames and return dictionary of face_id -> frames"""
        face_frames = {}  # face_id -> list of frames

        # Load mean face landmarks if not already loaded
        if not hasattr(self, "mean_face_landmarks"):
            mean_face_path = "20words_mean_face.npy"  # You'll need this file
            self.mean_face_landmarks = np.load(mean_face_path)

        # Initialize transforms
        to_tensor = transforms.ToTensor()
        to_gray = transforms.Grayscale()

        # Use deque for landmark smoothing
        q_landmarks = deque(maxlen=self.config.window_margin)

        # Pre-detect faces for all frames in batch
        batch_boxes = []
        batch_scores = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.numpy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            boxes, scores = self.face_detector.detect(frame)
            batch_boxes.append(boxes if boxes is not None else np.array([]))
            batch_scores.append(scores if scores is not None else np.array([]))

        prev_boxes = None
        face_id_counter = 0
        current_face_ids = []

        # Process each frame
        for frame_idx, (frame, boxes, scores) in enumerate(
            zip(frames, batch_boxes, batch_scores)
        ):
            try:
                if len(boxes) == 0:
                    continue

                # Match current faces with previous faces and assign IDs
                if prev_boxes is not None:
                    current_face_ids = [-1] * len(boxes)
                    for i, curr_box in enumerate(boxes):
                        ious = np.array(
                            [
                                self._calculate_iou(curr_box, prev_box)
                                for prev_box in prev_boxes.values()
                            ]
                        )
                        if len(ious) > 0:
                            max_iou_idx = np.argmax(ious)
                            max_iou = ious[max_iou_idx]
                            if max_iou > 0.5:
                                current_face_ids[i] = list(prev_boxes.keys())[
                                    max_iou_idx
                                ]

                    for i in range(len(boxes)):
                        if current_face_ids[i] == -1:
                            current_face_ids[i] = face_id_counter
                            face_id_counter += 1
                else:
                    current_face_ids = list(range(len(boxes)))
                    face_id_counter = len(boxes)

                prev_boxes = {
                    face_id: box for face_id, box in zip(current_face_ids, boxes)
                }

                # Process each detected face
                for face_idx, (box, score) in enumerate(zip(boxes, scores)):
                    if score < 0.9:
                        continue

                    # Get face crop with margin
                    x1, y1, x2, y2 = map(int, box[:4])
                    w, h = x2 - x1, y2 - y1
                    margin_x = int(w * self.config.face_margin_factor)
                    margin_y = int(h * self.config.face_margin_factor)
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(frame.shape[1], x2 + margin_x)
                    y2 = min(frame.shape[0], y2 + margin_y)
                    face_crop = frame[y1:y2, x1:x2]

                    # Get landmarks
                    landmarks = self.landmark_detector.get_landmarks(face_crop)
                    if landmarks is None:
                        continue

                    # Add landmarks to queue for smoothing
                    q_landmarks.append(landmarks[0])

                    if len(q_landmarks) == self.config.window_margin:
                        # Get smoothed landmarks
                        smoothed_landmarks = np.mean(q_landmarks, axis=0)

                        # Align face using stable points
                        warped_face, transform = self._warp_frame(
                            face_crop,
                            smoothed_landmarks[self.config.stable_points],
                            self.mean_face_landmarks[self.config.stable_points],
                        )

                        # Transform landmarks to aligned face space
                        aligned_landmarks = transform(smoothed_landmarks)

                        # Crop mouth region
                        mouth_landmarks = aligned_landmarks[
                            self.config.start_idx : self.config.stop_idx
                        ]
                        mouth_center = np.mean(mouth_landmarks, axis=0)

                        half_size = self.config.crop_height // 2
                        start_x = int(max(0, mouth_center[0] - half_size))
                        end_x = int(
                            min(self.config.std_size[0], mouth_center[0] + half_size)
                        )
                        start_y = int(max(0, mouth_center[1] - half_size))
                        end_y = int(
                            min(self.config.std_size[1], mouth_center[1] + half_size)
                        )

                        mouth_crop = warped_face[start_y:end_y, start_x:end_x]

                        # Convert to grayscale and tensor
                        mouth_pil = Image.fromarray(mouth_crop)
                        mouth_pil = to_gray(mouth_pil)
                        mouth_tensor = to_tensor(mouth_pil)

                        # Store processed frame
                        face_id = current_face_ids[face_idx]
                        if face_id not in face_frames:
                            face_frames[face_id] = []
                        face_frames[face_id].append(mouth_tensor)

            except Exception as e:
                self.logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                continue

        return face_frames

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = map(int, box1[:4])
        x1_2, y1_2, x2_2, y2_2 = map(int, box2[:4])

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _visualize_debug(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        mouth_crop: np.ndarray,
        step: str,
    ):
        """Save debug visualizations"""
        if not self.config.debug_mode:
            return

        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(12, 4))

        # Plot original frame with landmarks
        plt.subplot(1, 2, 1)
        plt.imshow(frame)
        plt.plot(landmarks[:, 0], landmarks[:, 1], "r.")
        # Highlight mouth landmarks
        plt.plot(
            landmarks[self.config.start_idx : self.config.stop_idx, 0],
            landmarks[self.config.start_idx : self.config.stop_idx, 1],
            "g.",
        )
        plt.title("Face with Landmarks")

        # Plot cropped mouth region
        plt.subplot(1, 2, 2)
        if len(mouth_crop.shape) == 2:
            plt.imshow(mouth_crop, cmap="gray")
        else:
            plt.imshow(mouth_crop)
        plt.title(f"Cropped Mouth Region {mouth_crop.shape}")

        plt.tight_layout()
        plt.savefig(debug_dir / f"debug_{step}.png")
        plt.close()

    def _warp_frame(
        self, frame: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray
    ) -> Tuple[np.ndarray, tf.ProjectiveTransform]:
        """Warp frame to align with mean face landmarks using similarity transform"""
        tform = tf.estimate_transform("similarity", src_points, dst_points)
        warped = tf.warp(
            frame, inverse_map=tform.inverse, output_shape=self.config.std_size
        )
        warped = (warped * 255).astype(np.uint8)
        return warped, tform

    def _crop_mouth_region(
        self, frame: np.ndarray, mouth_landmarks: np.ndarray
    ) -> np.ndarray:
        """Crop mouth region from frame"""
        center_x, center_y = np.mean(mouth_landmarks, axis=0)
        half_height = self.config.crop_height // 2
        half_width = self.config.crop_width // 2

        start_y = int(center_y - half_height)
        end_y = int(center_y + half_height)
        start_x = int(center_x - half_width)
        end_x = int(center_x + half_width)

        # Ensure crop boundaries are within frame
        start_y = max(0, start_y)
        end_y = min(frame.shape[0], end_y)
        start_x = max(0, start_x)
        end_x = min(frame.shape[1], end_x)

        return frame[start_y:end_y, start_x:end_x]

    def _create_clips(
        self, frames: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """Create clips from processed frames"""
        clips = []
        lengths = []

        for i in range(
            0,
            len(frames) - self.config.frames_per_clip + 1,
            self.config.frames_per_clip,
        ):
            # Get frames for this clip
            clip_frames = frames[i : i + self.config.frames_per_clip]

            if len(clip_frames) == self.config.frames_per_clip:
                # Stack frames into a single tensor [T, C, H, W]
                clip = torch.stack(clip_frames)

                # Add batch dimension and rearrange to [C, T, H, W]
                clip = clip.permute(1, 0, 2, 3)
                clip = clip.unsqueeze(0)  # Add batch dimension [B, C, T, H, W]

                clips.append(clip)
                lengths.append(
                    [self.config.frames_per_clip]
                )  # Wrap in list for correct format

        return clips, lengths

    def _run_inference(
        self, clips: List[torch.Tensor], lengths: List[List[int]]
    ) -> float:
        """Run inference using ONNX Runtime"""
        if not self.ort_session:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Convert PyTorch tensor to numpy array
            clip = clips[0].cpu().numpy()  # Shape: [1, C, T, H, W]
            length = np.array(lengths[0], dtype=np.int64)

            # Get input names from the ONNX model
            input_name_x = self.ort_session.get_inputs()[0].name
            input_name_lengths = self.ort_session.get_inputs()[1].name

            # Run inference
            ort_inputs = {input_name_x: clip, input_name_lengths: length}
            ort_outputs = self.ort_session.run(None, ort_inputs)

            # Apply sigmoid to output
            prob = 1 / (1 + np.exp(-ort_outputs[0]))
            return float(prob.item())
        except Exception as e:
            self.logger.warning(f"Inference failed: {str(e)}")
            return None


# ============= Usage Example =============
def main():
    # Create configuration
    config = ProcessingConfig(debug_mode=True)

    # Initialize processor
    processor = VideoProcessor(config)

    # Load ONNX model
    processor.load_model("lipforensics.onnx")  # Changed to load ONNX model

    # Process video
    video_path = "Deepfake - Morgon Freeman.mp4"
    try:
        face_probabilities = processor.process_video(video_path)
        for face_id, probability in face_probabilities.items():
            print(f"Face {face_id} - Probability of being fake: {probability:.4f}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")


if __name__ == "__main__":
    main()
