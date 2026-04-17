"""
Video processing for YOLOv8 safety detection inference.

Phase 3: Load videos, extract frames, and handle streaming for inference.
Supports:
  • Video files — MP4, AVI, MOV, and other OpenCV-compatible formats
  • Live camera — webcam device index (0, 1, …) or RTSP/HTTP stream URL
"""

import os
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Union
import logging


logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process video files for inference."""

    def __init__(self, video_path: str, target_fps: int = 30, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file (MP4, AVI, MOV, etc.)
            target_fps: Resample video to this FPS (default: 30)
            target_size: Target frame size (height, width), default keeps original
        """
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.target_size = target_size

        # Validate file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Open video
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame skip for target FPS
        self.frame_skip = max(1, int(self.fps / self.target_fps))
        self.effective_fps = self.fps / self.frame_skip

        logger.info(f"Opened video: {self.video_path}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps} → {self.effective_fps} (skip={self.frame_skip})")
        logger.info(f"  Frames: {self.frame_count} ({self.frame_count / self.fps:.1f}s)")

    def __len__(self) -> int:
        """Number of frames after resampling to target FPS."""
        return self.frame_count // self.frame_skip

    def __iter__(self) -> Generator[Tuple[np.ndarray, int, Dict], None, None]:
        """
        Iterate through video frames.

        Yields:
            (frame, frame_idx, metadata) where:
            - frame: RGB numpy array (H, W, 3)
            - frame_idx: Frame index (0-based, after resampling)
            - metadata: Dict with timestamp, original_frame_id, etc.
        """
        frame_idx = 0
        actual_frame_idx = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Apply frame skip
            if actual_frame_idx % self.frame_skip != 0:
                actual_frame_idx += 1
                continue

            # Resize if needed
            if self.target_size:
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))

            # Convert BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create metadata
            metadata = {
                'frame_idx': frame_idx,
                'actual_frame_idx': actual_frame_idx,
                'timestamp_ms': (frame_idx / self.effective_fps) * 1000,
                'original_fps': self.fps,
                'effective_fps': self.effective_fps,
                'shape': frame.shape,
            }

            yield frame, frame_idx, metadata

            frame_idx += 1
            actual_frame_idx += 1

    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Get specific frame by index.

        Args:
            frame_idx: Frame index (0-based, after resampling)

        Returns:
            (frame, metadata)
        """
        actual_frame_idx = frame_idx * self.frame_skip
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise IndexError(f"Frame {frame_idx} out of range")

        # Resize if needed
        if self.target_size:
            frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        metadata = {
            'frame_idx': frame_idx,
            'actual_frame_idx': actual_frame_idx,
            'timestamp_ms': (frame_idx / self.effective_fps) * 1000,
        }

        return frame, metadata

    def get_slice(self, start_frame: int, end_frame: int) -> Generator[Tuple[np.ndarray, int, Dict], None, None]:
        """
        Get slice of video frames.

        Args:
            start_frame: Start frame index (inclusive)
            end_frame: End frame index (exclusive)

        Yields:
            (frame, frame_idx, metadata)
        """
        for frame_idx, (frame, idx, metadata) in enumerate(self):
            if idx >= end_frame:
                break
            if idx >= start_frame:
                yield frame, idx, metadata

    def close(self):
        """Release video resource."""
        self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def info(self) -> Dict:
        """Get video information."""
        return {
            'path': str(self.video_path),
            'resolution': (self.width, self.height),
            'fps': self.fps,
            'effective_fps': self.effective_fps,
            'frame_count': self.frame_count,
            'duration_seconds': self.frame_count / self.fps,
            'target_fps': self.target_fps,
            'frame_skip': self.frame_skip,
            'target_size': self.target_size,
        }


class VideoBatcher:
    """Batch frames from video for efficient processing."""

    def __init__(self, video_processor: VideoProcessor, batch_size: int = 1):
        """
        Initialize batcher.

        Args:
            video_processor: VideoProcessor instance
            batch_size: Number of frames per batch
        """
        self.video_processor = video_processor
        self.batch_size = batch_size

    def __iter__(self) -> Generator[Tuple[np.ndarray, list], None, None]:
        """
        Iterate through batches.

        Yields:
            (batch_frames, batch_metadata) where:
            - batch_frames: (B, H, W, 3) numpy array
            - batch_metadata: List of metadata dicts
        """
        batch_frames = []
        batch_metadata = []

        for frame, frame_idx, metadata in self.video_processor:
            batch_frames.append(frame)
            batch_metadata.append(metadata)

            if len(batch_frames) == self.batch_size:
                batch_array = np.stack(batch_frames)
                yield batch_array, batch_metadata
                batch_frames = []
                batch_metadata = []

        # Yield remaining frames
        if batch_frames:
            batch_array = np.stack(batch_frames)
            yield batch_array, batch_metadata


class VideoWriter:
    """Write frames with annotations to output video."""

    def __init__(self, output_path: str, fps: float = 30, frame_size: Tuple[int, int] = (640, 480)):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Output frames per second
            frame_size: Output frame size (width, height)
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            frame_size
        )

        if not self.writer.isOpened():
            raise ValueError(f"Failed to open video writer: {self.output_path}")

        logger.info(f"Opened video writer: {self.output_path} ({frame_size[0]}x{frame_size[1]} @ {fps} FPS)")

    def write_frame(self, frame: np.ndarray):
        """
        Write single frame.

        Args:
            frame: Frame as numpy array (H, W, 3) in RGB format
        """
        # Convert RGB → BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def write_frames(self, frames: list):
        """
        Write multiple frames.

        Args:
            frames: List of frames (RGB numpy arrays)
        """
        for frame in frames:
            self.write_frame(frame)

    def close(self):
        """Release video writer."""
        self.writer.release()
        logger.info(f"Closed video writer: {self.output_path}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class CameraProcessor:
    """Live camera or RTSP stream processor.

    Provides the same iterator interface as VideoProcessor so downstream
    code (inference scripts, compliance checks) works unchanged.

    Usage:
        # Webcam (device 0)
        with CameraProcessor(0, target_fps=15) as cam:
            for frame, idx, meta in cam:
                ...

        # RTSP stream
        with CameraProcessor("rtsp://192.168.1.100:554/stream") as cam:
            ...
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        target_fps: int = 30,
        target_size: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ):
        """
        Args:
            source: Webcam device index (0, 1, …) or RTSP/HTTP URL string
            target_fps: Cap frame rate to this value (skips frames if camera is faster)
            target_size: Resize frames to (height, width); None keeps native resolution
            max_frames: Stop after this many yielded frames; None runs until closed
        """
        self.source = source
        self.target_fps = target_fps
        self.target_size = target_size
        self.max_frames = max_frames

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera source: {source!r}")

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps  = self.cap.get(cv2.CAP_PROP_FPS) or target_fps

        # effective_fps exposed for VideoWriter compatibility
        self.fps = native_fps
        self.effective_fps = min(target_fps, native_fps)
        self._min_interval = 1.0 / self.effective_fps

        logger.info(
            f"Opened camera: {source!r}  "
            f"{self.width}×{self.height} @ {native_fps:.0f} fps  "
            f"(target {target_fps} fps)"
        )

    # VideoProcessor compatibility shim
    def info(self) -> Dict:
        return {
            'source': str(self.source),
            'resolution': (self.width, self.height),
            'fps': self.fps,
            'effective_fps': self.effective_fps,
            'target_fps': self.target_fps,
            'target_size': self.target_size,
            'max_frames': self.max_frames,
        }

    def __iter__(self) -> Generator[Tuple[np.ndarray, int, Dict], None, None]:
        """Yield (frame, frame_idx, metadata) at the requested frame rate."""
        frame_idx = 0
        last_yield = 0.0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera read failed — stream ended or disconnected")
                break

            now = time.time()
            if now - last_yield < self._min_interval:
                continue  # rate-limit: drop frame
            last_yield = now

            if self.target_size:
                frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            metadata = {
                'frame_idx': frame_idx,
                'timestamp_ms': now * 1000,
                'source': str(self.source),
                'shape': frame.shape,
            }

            yield frame, frame_idx, metadata

            frame_idx += 1
            if self.max_frames and frame_idx >= self.max_frames:
                break

    def close(self):
        """Release camera resource."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def validate_video(video_path: str) -> bool:
    """
    Quick validation that video can be opened.

    Args:
        video_path: Path to video file

    Returns:
        True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        cap.release()
        return True
    except Exception:
        return False


def get_video_info(video_path: str) -> Dict:
    """
    Get video properties without full processing.

    Args:
        video_path: Path to video file

    Returns:
        Dict with video properties
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info
