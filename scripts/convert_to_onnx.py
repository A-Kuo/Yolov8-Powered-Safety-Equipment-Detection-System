#!/usr/bin/env python3

"""
Convert YOLOv8 PyTorch models to ONNX format.

Usage:
    python scripts/convert_to_onnx.py \
        --model models/yolo/worker_detector.pt \
        --output models/onnx/worker_detector.onnx
"""

import argparse
import logging
from pathlib import Path

from src.inference import YOLODetector
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv8 models to ONNX")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--output", help="Output ONNX path (auto-generated if not specified)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    setup_logging(log_level="INFO")

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    output_path = args.output or str(model_path.with_suffix(".onnx"))

    logger.info(f"Converting {model_path} to ONNX...")
    logger.info(f"Output: {output_path}")

    try:
        detector = YOLODetector(str(model_path), device="cpu")
        detector.export_onnx(output_path, opset=args.opset)
        logger.info("✓ Conversion successful")
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
