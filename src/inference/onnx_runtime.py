"""ONNX Runtime Inference Engine

Provides inference using ONNX Runtime with support for QNN execution provider
on Qualcomm devices (Rubik Pi, Snapdragon).
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class ONNXInference:
    """ONNX Runtime inference wrapper with QNN support."""

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        use_qnn: bool = False,
    ):
        """Initialize ONNX Runtime session.

        Args:
            model_path: Path to ONNX model (.onnx)
            providers: List of execution providers
                      (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            use_qnn: Use QNN provider if available (Qualcomm devices)

        Raises:
            ImportError: If onnxruntime not installed
            FileNotFoundError: If model_path does not exist
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Set execution providers
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        if use_qnn:
            providers.insert(0, "QNNExecutionProvider")

        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            logger.info(f"Loaded ONNX model from {model_path}")
            logger.info(f"Using providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on image.

        Args:
            image: Input image, shape should match model's expected input

        Returns:
            Dictionary with output names as keys and numpy arrays as values
        """
        # Ensure image is in correct format (typically NCHW)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        # Prepare input
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        # Run inference
        inputs = {self.input_name: image}
        outputs = self.session.run(self.output_names, inputs)

        # Return as dictionary
        return {name: output for name, output in zip(self.output_names, outputs)}

    def get_input_details(self) -> Dict[str, Any]:
        """Get input layer details."""
        return {
            "name": self.input_name,
            "shape": self.input_shape,
            "type": str(self.session.get_inputs()[0].type),
        }

    def get_output_details(self) -> List[Dict[str, Any]]:
        """Get output layer details."""
        outputs = []
        for output in self.session.get_outputs():
            outputs.append(
                {
                    "name": output.name,
                    "shape": output.shape,
                    "type": str(output.type),
                }
            )
        return outputs
