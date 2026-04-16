"""Inference module for YOLOv8 models"""

from .yolo_detector import YOLODetector
from .onnx_runtime import ONNXInference

__all__ = ["YOLODetector", "ONNXInference"]
