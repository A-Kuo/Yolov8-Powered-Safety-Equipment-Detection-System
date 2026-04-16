"""Tests for inference modules"""

import pytest
import numpy as np
from pathlib import Path


class TestYOLODetector:
    """Test YOLODetector class"""

    @pytest.mark.skip(reason="Requires model checkpoint")
    def test_detector_initialization(self):
        """Test that YOLODetector initializes without errors."""
        from src.inference import YOLODetector

        detector = YOLODetector(
            model_path="models/yolo/worker_detector.pt",
            device="cpu",
            conf=0.5,
        )
        assert detector is not None

    @pytest.mark.skip(reason="Requires model checkpoint")
    def test_predict_shape(self, sample_image):
        """Test that predict returns expected output shape."""
        from src.inference import YOLODetector

        detector = YOLODetector(
            model_path="models/yolo/worker_detector.pt",
            device="cpu",
        )
        result = detector.predict(sample_image)

        assert "boxes" in result
        assert "confidences" in result
        assert "class_ids" in result
        assert len(result["boxes"].shape) == 2
        assert result["boxes"].shape[1] == 4


class TestONNXInference:
    """Test ONNXInference class"""

    def test_onnx_import(self):
        """Test that onnxruntime can be imported."""
        try:
            import onnxruntime
            assert onnxruntime is not None
        except ImportError:
            pytest.skip("onnxruntime not installed")

    @pytest.mark.skip(reason="Requires ONNX model checkpoint")
    def test_onnx_initialization(self):
        """Test ONNXInference initialization."""
        from src.inference import ONNXInference

        engine = ONNXInference(
            model_path="models/onnx/worker_detector.onnx",
            providers=["CPUExecutionProvider"],
        )
        assert engine is not None

    @pytest.mark.skip(reason="Requires ONNX model checkpoint")
    def test_onnx_predict(self, sample_image):
        """Test ONNX inference."""
        from src.inference import ONNXInference

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

    def test_nms(self):
        """Test Non-Maximum Suppression."""
        from src.inference.postprocess import nms

        # Two overlapping boxes with high IOU
        boxes = np.array([
            [10, 10, 100, 100],
            [20, 20, 110, 110],  # ~80% overlap
            [200, 200, 300, 300],  # Far away
        ], dtype=np.float32)

        confidences = np.array([0.9, 0.7, 0.8])

        keep_indices = nms(boxes, confidences, iou_threshold=0.5)

        # Should keep indices 0 (highest conf) and 2 (far away)
        assert len(keep_indices) == 2
        assert 0 in keep_indices
        assert 2 in keep_indices
