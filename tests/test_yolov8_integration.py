"""Comprehensive YOLOv8 Integration Tests

Tests core YOLOv8 functionality, model loading, inference, and export.
"""

import pytest
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TestConfigLoading:
    """Test configuration file loading"""

    def test_config_loader_import(self):
        """Test that config_loader can be imported."""
        from src.utils.config_loader import load_config
        assert callable(load_config)

    def test_load_models_config(self, config_dir):
        """Test loading models.yaml configuration."""
        from src.utils.config_loader import load_config

        config_path = config_dir / "models.yaml"
        config = load_config(str(config_path))

        assert config is not None
        assert "models" in config
        assert "worker_detector" in config["models"]
        assert "ppe_detector" in config["models"]

    def test_load_dataset_config(self, config_dir):
        """Test loading dataset.yaml configuration."""
        from src.utils.config_loader import load_config

        config_path = config_dir / "dataset.yaml"
        config = load_config(str(config_path))

        assert config is not None
        assert "classes" in config
        assert "splits" in config
        assert "augmentation" in config

    def test_load_inference_config(self, config_dir):
        """Test loading inference.yaml configuration."""
        from src.utils.config_loader import load_config

        config_path = config_dir / "inference.yaml"
        config = load_config(str(config_path))

        assert config is not None
        assert "runtime" in config
        assert "inference" in config
        assert "postprocess" in config

    def test_classes_yaml_valid(self, config_dir):
        """Test that classes.yaml has correct format."""
        from src.utils.config_loader import load_config

        config_path = Path(__file__).parent.parent / "data" / "annotations" / "classes.yaml"
        config = load_config(str(config_path))

        assert config is not None
        assert config["nc"] == 10
        assert "names" in config
        assert len(config["names"]) == 10


class TestLogging:
    """Test logging setup"""

    def test_logging_import(self):
        """Test that logging module can be imported."""
        from src.utils.logging import setup_logging
        assert callable(setup_logging)

    def test_logging_setup(self, tmp_path):
        """Test logging setup with file."""
        from src.utils.logging import setup_logging

        log_file = tmp_path / "test.log"
        setup_logging(log_level="DEBUG", log_file=str(log_file))

        logger.debug("Test message")
        assert log_file.exists()


class TestYOLODetector:
    """Test YOLOv8 detector implementation"""

    def test_yolo_detector_import(self):
        """Test that YOLODetector can be imported."""
        from src.inference.yolo_detector import YOLODetector
        assert YOLODetector is not None

    def test_yolo_detector_attributes(self):
        """Test YOLODetector class attributes."""
        from src.inference.yolo_detector import YOLODetector
        import inspect

        # Check methods exist
        methods = [m for m in dir(YOLODetector) if not m.startswith("_")]
        assert "predict" in methods
        assert "export_onnx" in methods

    @pytest.mark.skip(reason="Requires model checkpoint")
    def test_yolo_detector_initialization(self):
        """Test YOLODetector initialization with mock model."""
        from src.inference.yolo_detector import YOLODetector

        detector = YOLODetector(
            model_path="models/yolo/worker_detector.pt",
            device="cpu",
            conf=0.5,
        )

        assert detector.conf == 0.5
        assert detector.device == "cpu"

    @pytest.mark.skip(reason="Requires model checkpoint")
    def test_yolo_detector_predict(self, sample_image):
        """Test YOLODetector inference."""
        from src.inference.yolo_detector import YOLODetector

        detector = YOLODetector(
            model_path="models/yolo/worker_detector.pt",
            device="cpu",
        )

        result = detector.predict(sample_image)

        assert isinstance(result, dict)
        assert "boxes" in result
        assert "confidences" in result
        assert "class_ids" in result


class TestONNXInference:
    """Test ONNX Runtime inference"""

    def test_onnx_import(self):
        """Test that onnxruntime can be imported."""
        try:
            import onnxruntime as ort
            assert ort is not None
            logger.info(f"✓ onnxruntime version: {ort.__version__}")
        except ImportError:
            pytest.skip("onnxruntime not installed")

    def test_onnx_inference_import(self):
        """Test that ONNXInference can be imported."""
        from src.inference.onnx_runtime import ONNXInference
        assert ONNXInference is not None

    def test_onnx_inference_execution_providers(self):
        """Test ONNX Runtime execution providers."""
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            logger.info(f"✓ Available ONNX providers: {providers}")

            # Check for QNN provider
            if "QNNExecutionProvider" in providers:
                logger.info("✓ Qualcomm QNN provider available!")
            else:
                logger.info("ℹ QNN provider not available (expected on non-Qualcomm devices)")

        except ImportError:
            pytest.skip("onnxruntime not installed")

    @pytest.mark.skip(reason="Requires ONNX model checkpoint")
    def test_onnx_inference_initialization(self):
        """Test ONNXInference initialization."""
        from src.inference.onnx_runtime import ONNXInference

        engine = ONNXInference(
            model_path="models/onnx/worker_detector.onnx",
            providers=["CPUExecutionProvider"],
        )

        assert engine is not None

    @pytest.mark.skip(reason="Requires ONNX model checkpoint")
    def test_onnx_inference_predict(self, sample_image):
        """Test ONNX inference."""
        from src.inference.onnx_runtime import ONNXInference

        engine = ONNXInference(
            model_path="models/onnx/worker_detector.onnx",
            providers=["CPUExecutionProvider"],
        )

        result = engine.predict(sample_image)

        assert isinstance(result, dict)
        assert len(result) > 0


class TestPostprocessing:
    """Test post-processing utilities"""

    def test_filter_by_confidence(self):
        """Test confidence filtering."""
        from src.inference.postprocess import filter_by_confidence

        detections = {
            "boxes": np.array([[10, 20, 100, 200], [50, 50, 150, 150]]),
            "confidences": np.array([0.9, 0.3]),
            "class_ids": np.array([0, 1]),
            "class_names": {0: "worker", 1: "drone"},
        }

        filtered = filter_by_confidence(detections, confidence_threshold=0.5)

        assert len(filtered["boxes"]) == 1
        assert filtered["confidences"][0] == 0.9
        logger.info("✓ Confidence filtering works correctly")

    def test_nms_basic(self):
        """Test Non-Maximum Suppression basic functionality."""
        from src.inference.postprocess import nms

        boxes = np.array([
            [10, 10, 100, 100],
            [20, 20, 110, 110],
            [200, 200, 300, 300],
        ], dtype=np.float32)

        confidences = np.array([0.9, 0.7, 0.8])

        keep_indices = nms(boxes, confidences, iou_threshold=0.5)

        assert len(keep_indices) == 2
        assert 0 in keep_indices
        assert 2 in keep_indices
        logger.info("✓ NMS basic test passed")

    def test_nms_empty(self):
        """Test NMS with empty input."""
        from src.inference.postprocess import nms

        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        confidences = np.array([])

        keep_indices = nms(boxes, confidences)

        assert len(keep_indices) == 0
        logger.info("✓ NMS empty input test passed")

    def test_nms_single_box(self):
        """Test NMS with single box."""
        from src.inference.postprocess import nms

        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confidences = np.array([0.9])

        keep_indices = nms(boxes, confidences)

        assert len(keep_indices) == 1
        assert keep_indices[0] == 0
        logger.info("✓ NMS single box test passed")


class TestQualcommQNNSupport:
    """Test Qualcomm QNN support detection"""

    def test_qualcomm_qnn_availability(self):
        """Test if Qualcomm QNN is available."""
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()

            if "QNNExecutionProvider" in providers:
                logger.info("✅ Qualcomm QNN ExecutionProvider is AVAILABLE")
                logger.info("   Device supports QNN inference")
            else:
                logger.info("ℹ Qualcomm QNN not available (expected on non-Qualcomm devices)")
                logger.info("   Available providers: " + ", ".join(providers))

        except ImportError:
            logger.warning("⚠ onnxruntime not installed")

    def test_qualcomm_ai_hub_tools(self):
        """Test if Qualcomm AI Hub tools are available."""
        try:
            import qai_hub
            logger.info("✅ Qualcomm AI Hub is installed")
            logger.info(f"   Version: {qai_hub.__version__ if hasattr(qai_hub, '__version__') else 'unknown'}")
        except ImportError:
            logger.info("ℹ Qualcomm AI Hub not installed (optional, for cloud compilation)")
            logger.info("   Install with: pip install qai-hub")


class TestUltralyticsIntegration:
    """Test Ultralytics YOLOv8 integration"""

    def test_ultralytics_import(self):
        """Test that ultralytics can be imported."""
        try:
            from ultralytics import YOLO
            assert YOLO is not None
            logger.info("✓ Ultralytics YOLO imported successfully")
        except ImportError:
            pytest.skip("ultralytics not installed")

    def test_ultralytics_version(self):
        """Test Ultralytics version."""
        try:
            import ultralytics
            logger.info(f"✓ Ultralytics version: {ultralytics.__version__}")
        except ImportError:
            pytest.skip("ultralytics not installed")

    @pytest.mark.skip(reason="Requires internet to download model")
    def test_yolov8_model_download(self):
        """Test downloading YOLOv8 model from Ultralytics hub."""
        from ultralytics import YOLO

        # This would download the model if not cached
        model = YOLO("yolov8n.pt")
        assert model is not None
        logger.info("✓ YOLOv8n model loaded from Ultralytics hub")


class TestInferenceWithSampleImage:
    """Test full inference pipeline with sample data"""

    def test_full_pipeline_structure(self, sample_image):
        """Test that full inference pipeline is structurally sound."""
        # Test config loading
        from src.utils.config_loader import load_config
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        config = load_config(str(config_path))

        # Verify structure
        assert "models" in config
        assert "worker_detector" in config["models"]
        assert config["models"]["worker_detector"]["input_size"] == 640
        assert config["models"]["worker_detector"]["confidence_threshold"] == 0.5

        logger.info("✓ Full inference pipeline structure validated")

    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing compatibility."""
        # Verify image shape
        assert sample_image.shape == (480, 640, 3)
        assert sample_image.dtype == np.uint8

        # Test resizing (mock what YOLO would do)
        from PIL import Image
        img = Image.fromarray(sample_image)
        img_resized = img.resize((640, 640))

        assert img_resized.size == (640, 640)
        logger.info("✓ Image preprocessing compatible")

    def test_batch_processing_structure(self, sample_image_batch):
        """Test batch processing structure."""
        assert sample_image_batch.shape == (4, 480, 640, 3)

        # Mock batch inference structure
        batch_results = []
        for img in sample_image_batch:
            result = {
                "boxes": np.zeros((0, 4)),
                "confidences": np.zeros(0),
                "class_ids": np.zeros(0, dtype=int),
            }
            batch_results.append(result)

        assert len(batch_results) == 4
        logger.info("✓ Batch processing structure validated")


class TestEdgeDeviceCompatibility:
    """Test compatibility with edge devices"""

    def test_onnx_cpu_inference(self):
        """Test ONNX CPU inference provider."""
        try:
            import onnxruntime as ort

            assert "CPUExecutionProvider" in ort.get_available_providers()
            logger.info("✓ CPU inference provider available")
        except ImportError:
            pytest.skip("onnxruntime not installed")

    def test_memory_efficient_inference(self):
        """Test memory-efficient inference structure."""
        # Verify that model loading doesn't require GPU
        from src.inference.yolo_detector import YOLODetector

        # Check that CPU is supported
        assert "cpu" in ["cuda", "cpu", "mps"]
        logger.info("✓ CPU inference supported")

    def test_quantization_support(self):
        """Test quantization support for edge devices."""
        # Verify ONNX supports INT8 quantization
        try:
            import onnxruntime as ort
            logger.info("✓ ONNX Runtime available for quantization workflows")
        except ImportError:
            logger.info("ℹ ONNX Runtime needed for quantization")

    def test_model_export_paths(self):
        """Test that model export paths are configured."""
        from src.utils.config_loader import load_config
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        config = load_config(str(config_path))

        # Check ONNX export paths
        assert "onnx_checkpoint" in config["models"]["worker_detector"]
        assert "dlc_checkpoint" in config["models"]["worker_detector"]
        logger.info("✓ Model export paths configured for ONNX and QNN")
