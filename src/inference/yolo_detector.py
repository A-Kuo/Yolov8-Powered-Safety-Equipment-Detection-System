"""YOLOv8 Detection Wrapper

Provides a unified interface for YOLOv8 object detection using Ultralytics.
Supports multiple backends (PyTorch, ONNX, QNN).
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLOv8 detector wrapper for unified inference interface."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        half: bool = True,
        conf: float = 0.5,
        iou: float = 0.45,
    ):
        """Initialize YOLOv8 detector.

        Args:
            model_path: Path to model checkpoint (.pt)
            device: Device to use (cuda, cpu, mps)
            half: Use FP16 precision for faster inference
            conf: Confidence threshold for detections
            iou: IOU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.device = device
        self.half = half
        self.conf = conf
        self.iou = iou

        if not self.model_path.exists():
            logger.warning(
                f"Model path {model_path} does not exist. "
                "Will attempt to download from Ultralytics hub."
            )

        self.model = YOLO(str(model_path))
        self.model.to(device)
        logger.info(f"Loaded YOLOv8 model from {model_path} on {device}")

    def predict(
        self, image: np.ndarray, conf: Optional[float] = None, iou: Optional[float] = None
    ) -> Dict[str, Any]:
        """Run inference on image.

        Args:
            image: Input image (BGR, HxWx3 numpy array)
            conf: Confidence threshold (overrides default)
            iou: IOU threshold (overrides default)

        Returns:
            Dictionary containing:
                - boxes: Nx4 array of [x1, y1, x2, y2]
                - confidences: N array of confidence scores
                - class_ids: N array of class indices
                - class_names: List of class names
        """
        conf = conf or self.conf
        iou = iou or self.iou

        results = self.model(
            image,
            conf=conf,
            iou=iou,
            half=self.half,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        detections = {
            "boxes": result.boxes.xyxy.cpu().numpy(),
            "confidences": result.boxes.conf.cpu().numpy(),
            "class_ids": result.boxes.cls.cpu().numpy().astype(int),
            "class_names": result.names,
        }

        return detections

    def export_onnx(self, output_path: str, opset: int = 12) -> None:
        """Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            opset: ONNX opset version
        """
        logger.info(f"Exporting model to ONNX: {output_path}")
        self.model.export(format="onnx", opset=opset)
        logger.info(f"ONNX model saved to {output_path}")
