import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Union

import face_alignment
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
from skimage import transform as tf
from torchvision import transforms
from torchvision.io import read_video
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

import cv2

# ============= Helper Functions =============


def load_json(json_fp):
    with open(json_fp, "r") as f:
        json_content = json.load(f)
    return json_content


def reshape_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


def _average_batch(x, lengths):
    return torch.stack(
        [torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0
    )


def downsample_basic_block(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes),
    )


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


# ============= Model Definitions =============


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu"):
        super(BasicBlock, self).__init__()
        assert relu_type in ["relu", "prelu"]
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = (
            nn.PReLU(num_parameters=planes)
            if relu_type == "prelu"
            else nn.ReLU(inplace=True)
        )
        self.relu2 = (
            nn.PReLU(num_parameters=planes)
            if relu_type == "prelu"
            else nn.ReLU(inplace=True)
        )
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu2(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        relu_type="relu",
        gamma_zero=False,
        avg_pool_downsample=False,
    ):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes * block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, relu_type=self.relu_type)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=self.relu_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert (
                self.chomp_size % 2 == 0
            ), "If symmetric chomp, chomp size needs to be even"

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size // 2 : -self.chomp_size // 2].contiguous()
        else:
            return x[:, :, : -self.chomp_size].contiguous()


class ConvBatchChompRelu(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        relu_type,
        dwpw=False,
    ):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                nn.Conv1d(
                    n_inputs,
                    n_inputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=n_inputs,
                    bias=False,
                ),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                (
                    nn.PReLU(num_parameters=n_inputs)
                    if relu_type == "prelu"
                    else nn.ReLU(inplace=True)
                ),
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                (
                    nn.PReLU(num_parameters=n_outputs)
                    if relu_type == "prelu"
                    else nn.ReLU(inplace=True)
                ),
            )
        else:
            self.conv = nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding, True)
            self.non_lin = (
                nn.PReLU(num_parameters=n_outputs)
                if relu_type == "prelu"
                else nn.ReLU()
            )

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)


class MultibranchTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_sizes,
        stride,
        dilation,
        padding,
        dropout=0.2,
        relu_type="relu",
        dwpw=False,
    ):
        super(MultibranchTemporalBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        self.n_outputs_branch = n_outputs // self.num_kernels
        assert (
            n_outputs % self.num_kernels == 0
        ), "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(
                n_inputs,
                self.n_outputs_branch,
                k,
                stride,
                dilation,
                padding[k_idx],
                relu_type,
                dwpw=dwpw,
            )
            setattr(self, f"cbcr0_{k_idx}", cbcr)
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):
            cbcr = ConvBatchChompRelu(
                n_outputs,
                self.n_outputs_branch,
                k,
                stride,
                dilation,
                padding[k_idx],
                relu_type,
                dwpw=dwpw,
            )
            setattr(self, f"cbcr1_{k_idx}", cbcr)
        self.dropout1 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if (n_inputs // self.num_kernels) != n_outputs
            else None
        )
        self.relu_final = (
            nn.PReLU(num_parameters=n_outputs) if relu_type == "prelu" else nn.ReLU()
        )

    def forward(self, x):
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, f"cbcr0_{k_idx}")
            outputs.append(branch_convs(x))
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0(out0)

        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, f"cbcr1_{k_idx}")
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out1 + res)


class MultibranchTemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        tcn_options,
        dropout=0.2,
        relu_type="relu",
        dwpw=False,
    ):
        super(MultibranchTemporalConvNet, self).__init__()
        self.ksizes = tcn_options["kernel_size"]
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = [(s - 1) * dilation_size for s in self.ksizes]
            layers.append(
                MultibranchTemporalBlock(
                    in_channels,
                    out_channels,
                    self.ksizes,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                    relu_type=relu_type,
                    dwpw=dwpw,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiscaleMultibranchTCN(nn.Module):
    def __init__(
        self,
        input_size,
        num_channels,
        num_classes,
        tcn_options,
        dropout,
        relu_type,
        dwpw=False,
    ):
        super(MultiscaleMultibranchTCN, self).__init__()
        self.kernel_sizes = tcn_options["kernel_size"]
        self.num_kernels = len(self.kernel_sizes)
        self.mb_ms_tcn = MultibranchTemporalConvNet(
            input_size,
            num_channels,
            tcn_options,
            dropout=dropout,
            relu_type=relu_type,
            dwpw=dwpw,
        )
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        out = self.mb_ms_tcn(x)
        out = self.consensus_func(out, lengths)
        return self.tcn_output(out)


class Lipreading(nn.Module):
    def __init__(
        self, hidden_dim=256, num_classes=1, relu_type="prelu", tcn_options={}
    ):
        super(Lipreading, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = (
            nn.PReLU(num_parameters=self.frontend_nout)
            if relu_type == "prelu"
            else nn.ReLU()
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.tcn = MultiscaleMultibranchTCN(
            input_size=self.backend_out,
            num_channels=[
                hidden_dim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]
            ]
            * tcn_options["num_layers"],
            num_classes=num_classes,
            tcn_options=tcn_options,
            dropout=tcn_options["dropout"],
            relu_type=relu_type,
            dwpw=tcn_options["dwpw"],
        )

    def forward(self, x, lengths):
        x = self.frontend3D(x)
        t_new = x.shape[2]
        x = reshape_tensor(x)
        x = self.trunk(x)
        x = x.view(-1, t_new, x.size(1))
        return self.tcn(x, lengths)


# ============= Configuration =============
@dataclass
class ProcessingConfig:
    """Configuration for video processing parameters"""

    window_margin: int = 12
    crop_height: int = 96
    crop_width: int = 96
    frames_per_clip: int = 25
    face_margin_factor: float = 0.2
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
        self.forgery_model = None

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

    def load_model(self, weights_path: str):
        """Load the forgery detection model"""
        self.logger.info(f"Loading model weights from {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.forgery_model = self._build_model()
            self.forgery_model.load_state_dict(checkpoint["model"])
            self.forgery_model.to(self.device)
            self.forgery_model.eval()
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def _build_model(self):
        """Build the LipForensics model architecture"""
        # Load model configuration
        args_loaded = load_json("lrw_resnet18_mstcn.json")
        relu_type = args_loaded["relu_type"]
        tcn_options = {
            "num_layers": args_loaded["tcn_num_layers"],
            "kernel_size": args_loaded["tcn_kernel_size"],
            "dropout": args_loaded["tcn_dropout"],
            "dwpw": args_loaded["tcn_dwpw"],
            "width_mult": args_loaded["tcn_width_mult"],
        }

        # Create Lipreading model
        model = Lipreading(
            hidden_dim=256,
            num_classes=1,  # Binary classification for real/fake
            relu_type=relu_type,
            tcn_options=tcn_options,
        )

        return model

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

    def process_video(self, video_path: Union[str, Path]) -> float:
        """Process a video file and return probability of being fake"""
        try:
            self.logger.info(f"Processing video: {video_path}")

            # Process frames in batches
            batch_size = self.config.frames_per_clip
            current_batch = []
            all_processed_frames = []

            # Use generator for frame loading
            for frame_idx, frame in enumerate(self._load_video_frames(video_path)):
                current_batch.append(frame)

                # Process when batch is full or at end of video
                if len(current_batch) == batch_size:
                    # Process batch of frames
                    landmarks = self._process_landmarks(np.array(current_batch))
                    processed_frames = self._process_mouth_regions(
                        np.array(current_batch), landmarks
                    )
                    all_processed_frames.extend(processed_frames)

                    # Clear batch
                    current_batch = []

                # Optional: limit total frames for debugging
                if frame_idx >= 250:  # Adjust as needed
                    break

            # Process any remaining frames
            if current_batch:
                landmarks = self._process_landmarks(np.array(current_batch))
                processed_frames = self._process_mouth_regions(
                    np.array(current_batch), landmarks
                )
                all_processed_frames.extend(processed_frames)

            # Create clips and run inference
            clips, lengths = self._create_clips(all_processed_frames)
            probability = self._run_inference(clips, lengths)

            self.logger.info(f"Inference complete. Probability: {probability:.4f}")
            return probability

        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            raise

    def _process_landmarks(self, frames: torch.Tensor) -> List[np.ndarray]:
        """Process frames to extract facial landmarks"""
        landmarks_list = []

        for frame in tqdm(frames, desc="Processing landmarks"):
            try:
                # Convert frame format
                frame_np = frame.numpy()

                # Detect face
                boxes, _ = self.face_detector.detect(frame_np)
                if boxes is None or len(boxes) == 0:
                    self.logger.warning("No face detected in frame")
                    continue

                # Get face crop
                box = boxes[0]
                margin = 20
                x1, y1, x2, y2 = map(int, box[:4])
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame_np.shape[1], x2 + margin)
                y2 = min(frame_np.shape[0], y2 + margin)
                face_crop = frame_np[y1:y2, x1:x2]

                # Get landmarks
                landmarks = self.landmark_detector.get_landmarks(face_crop)
                if landmarks is None:
                    self.logger.warning("No landmarks detected in face")
                    continue

                landmarks_list.append(landmarks[0])

                # if self.config.debug_mode:
                #     self._visualize_debug(frame_np, landmarks[0], face_crop, f"frame_{len(landmarks_list)}")

            except Exception as e:
                self.logger.warning(f"Error processing frame landmarks: {str(e)}")
                continue

        if not landmarks_list:
            raise ValueError("No valid landmarks detected in video")

        return landmarks_list

    def _process_mouth_regions(
        self, frames: torch.Tensor, landmarks: List[np.ndarray]
    ) -> List[torch.Tensor]:
        """Extract and process mouth regions from frames"""
        processed_frames = []

        margin_factor = self.config.face_margin_factor

        # Initialize transforms
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        to_gray = transforms.Grayscale()
        resize_face = transforms.Resize((256, 256))

        for frame in tqdm(frames, desc="Processing frames"):
            # Ensure frame is in the correct format (H, W, C)
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0)
                frame_np = frame.numpy()
            else:
                frame_np = frame

            # Convert to uint8 if needed
            if frame_np.dtype != np.uint8:
                frame_np = (frame_np * 255).astype(np.uint8)

            # Detect face
            boxes, _ = self.face_detector.detect(frame_np)
            if boxes is None or len(boxes) == 0:
                continue

            # Get face crop coordinates with margin factor
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1

            # Calculate margins based on face size
            x1_margin = max(0, int(x1 - margin_factor * w))
            y1_margin = max(0, int(y1 - margin_factor * h))
            x2_margin = min(frame_np.shape[1], int(x2 + margin_factor * w))
            y2_margin = min(frame_np.shape[0], int(y2 + margin_factor * h))

            # Crop face with margins
            face_crop = frame_np[y1_margin:y2_margin, x1_margin:x2_margin]

            # Get landmarks for cropped face
            face_landmarks = self.landmark_detector.get_landmarks(face_crop)
            if face_landmarks is None:
                continue

            # Convert cropped face to PIL and resize
            face_pil = Image.fromarray(face_crop)
            face_pil = resize_face(face_pil)
            face_np = np.array(face_pil)

            # Scale landmarks to match resized face
            scale_x = 256 / face_crop.shape[1]
            scale_y = 256 / face_crop.shape[0]
            landmarks_scaled = face_landmarks[0].copy()
            landmarks_scaled[:, 0] *= scale_x
            landmarks_scaled[:, 1] *= scale_y

            # Get mouth landmarks and center
            mouth_landmarks = landmarks_scaled[
                self.config.start_idx : self.config.stop_idx
            ]
            mouth_center = np.mean(mouth_landmarks, axis=0)

            # Crop 96x96 around mouth center
            half_size = 48  # 96/2
            start_x = int(max(0, mouth_center[0] - half_size))
            end_x = int(min(256, mouth_center[0] + half_size))
            start_y = int(max(0, mouth_center[1] - half_size))
            end_y = int(min(256, mouth_center[1] + half_size))

            # Crop mouth region
            mouth_crop = face_np[start_y:end_y, start_x:end_x]

            # Convert to grayscale and tensor
            mouth_pil = Image.fromarray(mouth_crop)
            mouth_pil = to_gray(mouth_pil)
            mouth_tensor = to_tensor(mouth_pil)

            # Add debug logging
            self.logger.debug(f"Frame shape: {frame_np.shape}")
            self.logger.debug(f"Face crop shape: {face_crop.shape}")
            self.logger.debug(f"Resized face shape: {face_np.shape}")
            self.logger.debug(f"Mouth crop shape: {mouth_crop.shape}")
            self.logger.debug(f"Mouth tensor shape: {mouth_tensor.shape}")

            processed_frames.append(mouth_tensor)

            if self.config.debug_mode:
                self._visualize_debug(
                    face_np,
                    landmarks_scaled,
                    np.array(mouth_pil),
                    f"frame_{len(processed_frames)}",
                )

        if not processed_frames:
            raise ValueError("No valid mouth regions were processed")

        return self._create_clips(processed_frames)

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
        """Warp frame to align with mean face landmarks"""
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

    def _create_clips(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Create clips from processed frames"""
        clips = []
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

        return clips, [[self.config.frames_per_clip]] * len(clips)

    def _run_inference(self, clips: List[torch.Tensor], lengths) -> float:
        """Run inference on processed clips"""
        if not self.forgery_model:
            raise ValueError("Model not loaded. Call load_model() first.")

        probabilities = []
        with torch.no_grad():
            for clip, length in zip(clips, lengths):
                clip = clip.to(self.device)
                output = self.forgery_model(clip, lengths=length)
                prob = torch.sigmoid(output).item()
                probabilities.append(prob)

        return np.mean(probabilities)


# ============= Usage Example =============
def main():
    # Create configuration
    config = ProcessingConfig(debug_mode=True)

    # Initialize processor
    processor = VideoProcessor(config)

    # Load model weights
    processor.load_model("lipforensics_ff.pth")

    # Process video
    video_path = "046_W010.mp4"
    try:
        probability = processor.process_video(video_path)
        print(f"Probability of being fake: {probability:.4f}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")


if __name__ == "__main__":
    main()
