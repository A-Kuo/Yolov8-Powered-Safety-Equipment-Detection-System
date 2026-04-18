#!/usr/bin/env python
"""
Basic inference example for PPE detection system.

This example demonstrates how to use the YOLODetector for real-time inference.
It works with the pre-trained models in the repository.
"""

import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.yolo_detector import YOLODetector
from src.utils.config_loader import load_config
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def example_synthetic_detection():
    """Example 1: Run inference with synthetic image data."""
    logger.info("=" * 60)
    logger.info("Example 1: Synthetic Image Inference")
    logger.info("=" * 60)

    # Create synthetic test image (640x640 RGB)
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Check if model exists
    model_path = "models/yolo/ppe_detector.pt"
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}")
        logger.info("To get started, download models or train your own:")
        logger.info("  - See SETUP.md for setup instructions")
        logger.info("  - See PROJECT_STATUS.md for model availability")
        return

    # Initialize detector
    try:
        logger.info(f"Loading model from {model_path}")
        detector = YOLODetector(model_path, device="cpu", conf=0.5, iou=0.45)

        # Run inference
        logger.info("Running inference on test image...")
        results = detector.predict(test_image)

        # Display results
        logger.info(f"Detection complete:")
        logger.info(f"  - Objects detected: {len(results['boxes'])}")
        logger.info(f"  - Confidences: {results['confidences']}")
        logger.info(f"  - Classes: {results['class_names']}")

        if len(results['boxes']) > 0:
            logger.info(f"\nDetected objects:")
            for i, (box, conf, class_id) in enumerate(
                zip(results['boxes'], results['confidences'], results['class_ids'])
            ):
                class_name = results['class_names'][class_id]
                logger.info(f"  [{i}] {class_name}: conf={conf:.3f}, box={box}")
        else:
            logger.info("  No objects detected in test image")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        logger.info("This is expected if models are not available.")
        logger.info("See SETUP.md and PROJECT_STATUS.md for instructions.")


def example_load_config():
    """Example 2: Load and display configuration."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Configuration Loading")
    logger.info("=" * 60)

    config_path = "config/models.yaml"
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        return

    try:
        config = load_config(config_path)

        logger.info(f"Configuration loaded from {config_path}")
        if "models" in config:
            for model_name, model_config in config["models"].items():
                logger.info(f"\nModel: {model_name}")
                logger.info(f"  - Type: {model_config.get('type', 'N/A')}")
                logger.info(
                    f"  - Confidence threshold: {model_config.get('confidence_threshold', 'N/A')}"
                )
                logger.info(
                    f"  - Classes: {len(model_config.get('classes', []))} classes"
                )
        else:
            logger.warning("No 'models' section found in config")

    except Exception as e:
        logger.error(f"Error loading config: {e}")


def example_postprocessing():
    """Example 3: Demonstrate post-processing utilities."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Post-processing (NMS, Filtering)")
    logger.info("=" * 60)

    from src.inference.postprocess import nms, filter_by_confidence

    # Create sample detections
    boxes = np.array(
        [
            [10, 10, 50, 50],
            [15, 15, 55, 55],
            [100, 100, 150, 150],
            [200, 200, 250, 250],
        ]
    )
    confidences = np.array([0.9, 0.85, 0.5, 0.95])
    class_ids = np.array([0, 0, 1, 2])

    logger.info(f"Original detections: {len(boxes)} boxes")

    # Apply NMS
    keep_indices = nms(boxes, confidences, iou_threshold=0.5)
    logger.info(f"After NMS (IOU=0.5): {len(keep_indices)} boxes kept")

    # Apply confidence filtering
    detections = {
        "boxes": boxes,
        "confidences": confidences,
        "class_ids": class_ids,
        "class_names": {0: "person", 1: "vest", 2: "hat"},
    }
    filtered = filter_by_confidence(detections, confidence_threshold=0.6)
    logger.info(
        f"After confidence filter (0.6): {len(filtered['boxes'])} boxes kept"
    )


def main():
    """Run all examples."""
    logger.info("\n" + "🚀 " * 20)
    logger.info("YOLOv8 Safety Detection System - Examples")
    logger.info("🚀 " * 20 + "\n")

    try:
        example_load_config()
        example_postprocessing()
        example_synthetic_detection()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    logger.info("\n" + "✓ " * 20)
    logger.info("Examples complete!")
    logger.info("✓ " * 20)
    logger.info("\nNext steps:")
    logger.info("  1. See README.md for quick start guide")
    logger.info("  2. See SETUP.md for environment setup")
    logger.info("  3. Run 'pytest tests/ -v' to test the system")
    logger.info("  4. See PROJECT_STATUS.md for development roadmap")


if __name__ == "__main__":
    main()
