"""Video and Camera Input Processing

Handles:
- Webcam capture (USB, built-in)
- RTSP/HTTP streams
- Video file processing
- Frame rate control
- MP4 output writing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional, Union, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)


class CameraProcessor:
    """Live camera/RTSP stream capture with frame rate control."""

    def __init__(
        self,
        source: Union[int, str],
        target_fps: int = 30,
        target_size: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ):
        """Initialize camera processor.

        Args:
            source: Device index (0, 1, ...) or RTSP URL (rtsp://...)
            target_fps: Target frame rate (enforced via sleep)
            target_size: Optional resize target (width, height)
            max_frames: Max frames to capture (None = unlimited)
        """
        self.source = source
        self.target_fps = target_fps
        self.target_size = target_size
        self.max_frames = max_frames
        self._min_interval = 1.0 / target_fps if target_fps > 0 else 0

        self.cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera/stream: {source}")

        self.frame_count = 0
        self.last_time = time.time()
        logger.info(f"Opened camera/stream: {source} @ {target_fps} FPS")

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, Dict[str, Any]]]:
        """Iterator interface compatible with VideoProcessor."""
        while True:
            if self.max_frames and self.frame_count >= self.max_frames:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            elapsed = time.time() - self.last_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.last_time = time.time()

            if self.target_size:
                frame = cv2.resize(frame, self.target_size)

            metadata = {
                "timestamp_ms": int(self.frame_count * 1000 / self.target_fps),
                "source": str(self.source),
                "shape": frame.shape,
            }

            yield frame, self.frame_count, metadata
            self.frame_count += 1

    def close(self) -> None:
        if self.cap:
            self.cap.release()
        logger.info(f"Closed camera: {self.frame_count} frames")


class VideoProcessor:
    """Video file processing with FPS resampling."""

    def __init__(
        self,
        video_path: str,
        target_fps: int = 30,
        target_size: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ):
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.target_size = target_size
        self.max_frames = max_frames

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open: {video_path}")

        self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = 0
        self.native_frame_count = 0

        logger.info(f"Opened: {video_path} (native: {self.native_fps:.1f} FPS)")

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, Dict[str, Any]]]:
        frame_skip = max(1, int(self.native_fps / self.target_fps))

        while True:
            if self.max_frames and self.frame_count >= self.max_frames:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            if self.native_frame_count % frame_skip != 0:
                self.native_frame_count += 1
                continue

            if self.target_size:
                frame = cv2.resize(frame, self.target_size)

            metadata = {
                "timestamp_ms": int(self.frame_count * 1000 / self.target_fps),
                "source": str(self.video_path),
                "shape": frame.shape,
            }

            yield frame, self.frame_count, metadata
            self.frame_count += 1
            self.native_frame_count += 1

    def close(self) -> None:
        if self.cap:
            self.cap.release()
        logger.info(f"Closed video: {self.frame_count} frames")


class VideoWriter:
    """MP4 video output."""

    def __init__(
        self,
        output_path: str,
        frame_size: Tuple[int, int],
        fps: int = 30,
        codec: str = "mp4v",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            frame_size,
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {output_path}")

        logger.info(f"Video writer: {output_path} @ {fps} FPS")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer:
            self.writer.release()
        logger.info(f"Closed: {self.output_path}")
