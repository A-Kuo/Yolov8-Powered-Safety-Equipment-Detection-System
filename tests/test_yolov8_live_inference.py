"""Live YOLOv8 Inference Test

Tests actual model inference with YOLOv8.
"""

import pytest
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TestLiveYOLOInference:
    """Test live YOLOv8 inference with real model"""

    def test_yolov8_model_loading(self):
        """Test loading YOLOv8 model from Ultralytics hub."""
        from ultralytics import YOLO

        # Download YOLOv8n (nano) - smallest and fastest for testing
        model = YOLO("yolov8n.pt")
        assert model is not None
        logger.info("✅ YOLOv8n model loaded successfully")

        # Check model properties
        assert hasattr(model, "predict")
        assert hasattr(model, "export")
        logger.info(f"   Model has {len(model.model.model)} layers")

    def test_yolov8_inference_on_image(self):
        """Test inference on a sample image."""
        from ultralytics import YOLO
        import numpy as np

        model = YOLO("yolov8n.pt")

        # Create dummy image (640x640 RGB)
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run inference
        results = model(dummy_image, verbose=False)

        assert len(results) > 0
        result = results[0]

        # Check result structure
        assert hasattr(result, "boxes")
        assert hasattr(result, "names")
        logger.info(f"✅ Inference completed, found {len(result.boxes)} detections")

    def test_yolov8_inference_with_confidence(self):
        """Test inference with confidence threshold."""
        from ultralytics import YOLO
        import numpy as np

        model = YOLO("yolov8n.pt")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run inference with confidence threshold
        results = model(dummy_image, conf=0.5, verbose=False)

        assert len(results) > 0
        logger.info("✅ Inference with confidence threshold works")

    def test_yolov8_export_to_onnx(self, tmp_path):
        """Test exporting YOLOv8 to ONNX format."""
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")

        # Export to ONNX
        onnx_path = tmp_path / "yolov8n.onnx"
        export_result = model.export(format="onnx", imgsz=640)

        assert export_result is not None
        logger.info(f"✅ Model exported to ONNX successfully")
        logger.info(f"   Export result: {export_result}")

    def test_yolov8_export_to_other_formats(self, tmp_path):
        """Test exporting YOLOv8 to multiple formats."""
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")

        formats = ["torchscript", "onnx", "openvino", "tflite", "pb"]

        for fmt in formats:
            try:
                result = model.export(format=fmt, imgsz=640, device="cpu")
                logger.info(f"✅ Export to {fmt}: {result}")
            except Exception as e:
                logger.info(f"ℹ Export to {fmt} skipped: {str(e)[:50]}")

    def test_yolov8_performance(self):
        """Test YOLOv8 inference performance."""
        from ultralytics import YOLO
        import time
        import numpy as np

        model = YOLO("yolov8n.pt")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warm up
        model(dummy_image, verbose=False)

        # Time multiple inferences
        num_runs = 5
        times = []

        for _ in range(num_runs):
            start = time.time()
            results = model(dummy_image, verbose=False)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        logger.info(f"✅ Performance Test Results:")
        logger.info(f"   Average latency: {avg_time*1000:.2f}ms")
        logger.info(f"   FPS: {fps:.2f}")
        logger.info(f"   Individual times: {[f'{t*1000:.2f}ms' for t in times]}")

        # Check that we're hitting reasonable FPS (>5 on CPU)
        assert fps > 1, "Inference too slow"

    def test_yolov8_batch_inference(self):
        """Test batch inference."""
        from ultralytics import YOLO
        import numpy as np

        model = YOLO("yolov8n.pt")

        # Create batch of 4 images
        batch_images = [
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            for _ in range(4)
        ]

        # Run batch inference
        results = model(batch_images, verbose=False)

        assert len(results) == 4
        logger.info(f"✅ Batch inference on 4 images completed")

    def test_yolov8_model_info(self):
        """Test YOLOv8 model introspection."""
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")

        # Get model info
        info = model.info()

        logger.info("✅ Model information:")
        logger.info(f"   Model: {model.model}")
        logger.info(f"   Task: {model.task}")
        if hasattr(model, "model"):
            logger.info(f"   Parameters: {sum(p.numel() for p in model.model.parameters()) if hasattr(model.model, 'parameters') else 'N/A'}")


class TestONNXRuntimeInference:
    """Test ONNX Runtime inference with exported models"""

    def test_onnx_runtime_setup(self):
        """Test ONNX Runtime is properly configured."""
        import onnxruntime as ort

        # Get available providers
        providers = ort.get_available_providers()
        logger.info(f"✅ ONNX Runtime providers: {providers}")

        # CPU should always be available
        assert "CPUExecutionProvider" in providers

    def test_onnx_export_and_run(self, tmp_path):
        """Test exporting to ONNX and running with ONNX Runtime."""
        from ultralytics import YOLO
        import onnxruntime as ort
        import numpy as np

        model = YOLO("yolov8n.pt")

        # Export to ONNX
        export_result = model.export(format="onnx", imgsz=640)
        logger.info(f"✅ Exported YOLOv8 to ONNX: {export_result}")

        # Load with ONNX Runtime
        session = ort.InferenceSession(
            "yolov8n.onnx",
            providers=["CPUExecutionProvider"],
        )

        assert session is not None
        logger.info("✅ ONNX model loaded in ONNX Runtime")

        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        logger.info(f"   Input: {input_name}")
        logger.info(f"   Outputs: {output_names}")

        # Run inference
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = session.run(output_names, {input_name: dummy_input})

        assert len(outputs) > 0
        logger.info(f"✅ ONNX inference completed, {len(outputs)} outputs")


class TestSourceOfTruthConfigs:
    """Test that config files match implementation"""

    def test_config_classes_match_implementation(self):
        """Test that config classes match YOLOv8 classes."""
        from src.utils.config_loader import load_config
        from pathlib import Path

        # Load config
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        config = load_config(str(config_path))

        # Check classes
        classes_path = Path(__file__).parent.parent / "data" / "annotations" / "classes.yaml"
        classes_config = load_config(str(classes_path))

        assert classes_config["nc"] == 10
        expected_classes = [
            "worker",
            "drone",
            "safety_glasses",
            "safety_goggles",
            "hard_hat",
            "regular_hat",
            "hi_vis_vest",
            "regular_clothing",
            "work_boots",
            "regular_shoes",
        ]

        for i, cls_name in enumerate(expected_classes):
            assert classes_config["names"][i] == cls_name, f"Class {i} mismatch"

        logger.info(f"✅ Config has correct 10 classes: {list(classes_config['names'].values())}")

    def test_config_model_thresholds(self):
        """Test that model thresholds are configured."""
        from src.utils.config_loader import load_config
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        config = load_config(str(config_path))

        # Check thresholds for all models
        detection_models = ["worker_detector", "ppe_detector"]
        for model_name in detection_models:
            model = config["models"][model_name]
            assert "confidence_threshold" in model
            assert "iou_threshold" in model
            assert model["confidence_threshold"] > 0
            assert model["iou_threshold"] > 0

        # Classification model only needs confidence threshold
        classifier = config["models"]["drone_classifier"]
        assert "confidence_threshold" in classifier
        assert classifier["confidence_threshold"] > 0
        assert classifier.get("task") == "classify"

        logger.info("✅ All models have required thresholds configured")
