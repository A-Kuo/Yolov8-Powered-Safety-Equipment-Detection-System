"""
Multi-model inference pipeline for PPE detection.

Phase 3: Orchestrate worker detection → optional drone filtering → PPE detection.

Two backends are supported:
  • Roboflow Workflow (default when api_key provided) — cloud-based, no local models needed.
  • Local YOLOv8 models (fallback) — requires model files on disk.

Select backend via the `use_roboflow` constructor flag or by setting ROBOFLOW_API_KEY.

Optimizations available for local backend:
  • fp16=True      — FP16 half-precision (~1.5× speedup on supported GPUs)
  • input_size=480 — Reduce inference resolution (480 ≈ 1.5× faster, ~2% accuracy drop)
  • Batched PPE crops — all worker crops run in one YOLO call per frame
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from ultralytics import YOLO
from src.edge_deployment.safety_rules_engine import DetectionResult


logger = logging.getLogger(__name__)


class InferencePipeline:
    """Coordinate multi-model inference: worker detection → PPE detection.

    Set use_roboflow=True (or set ROBOFLOW_API_KEY env var) to call the
    Roboflow PPE Compliance Pipeline instead of running local YOLOv8 models.
    """

    def __init__(self,
                 worker_model_path: str = "",
                 ppe_model_path: str = "",
                 drone_model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 conf_threshold: float = 0.25,
                 use_roboflow: bool = False,
                 roboflow_api_key: Optional[str] = None,
                 fp16: bool = False,
                 input_size: int = 640):
        """
        Initialize inference pipeline.

        Args:
            worker_model_path: Path to YOLOv8 worker detector model (.pt)
            ppe_model_path: Path to YOLOv8 PPE detector model (.pt)
            drone_model_path: Optional path to drone classifier
            device: Device to use ('cuda', 'cpu', etc.)
            conf_threshold: Global confidence threshold
            use_roboflow: Use Roboflow cloud workflow instead of local models
            roboflow_api_key: Roboflow API key (overrides ROBOFLOW_API_KEY env var)
            fp16: Enable FP16 half-precision inference (~1.5× speedup, GPU only)
            input_size: Model input resolution (default 640; try 480 for 1.5× speedup)
        """
        self.device = device
        self.conf_threshold = conf_threshold
        self.fp16 = fp16 and device != 'cpu'
        self.input_size = input_size

        # Roboflow backend takes priority when requested or API key is available
        _api_key = roboflow_api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self._use_roboflow = use_roboflow or bool(_api_key)

        if self._use_roboflow:
            from src.edge_deployment.roboflow_inference import RoboflowInference
            self._roboflow = RoboflowInference(
                api_key=_api_key,
                conf_threshold=conf_threshold,
            )
            self.worker_model = None
            self.ppe_model = None
            self.drone_model = None
            self.worker_classes = {}
            self.ppe_classes = {}
            logger.info("✓ Using Roboflow workflow backend")
            return

        # Load local YOLO models
        logger.info(f"Loading models on {device} (fp16={self.fp16}, input_size={input_size})")

        self.worker_model = YOLO(worker_model_path)
        self.worker_model.to(device)
        if self.fp16:
            self.worker_model.model.half()
        logger.info(f"✓ Loaded worker detector: {worker_model_path}")

        self.ppe_model = YOLO(ppe_model_path)
        self.ppe_model.to(device)
        if self.fp16:
            self.ppe_model.model.half()
        logger.info(f"✓ Loaded PPE detector: {ppe_model_path}")

        self.drone_model = None
        if drone_model_path and Path(drone_model_path).exists():
            self.drone_model = YOLO(drone_model_path)
            self.drone_model.to(device)
            if self.fp16:
                self.drone_model.model.half()
            logger.info(f"✓ Loaded drone classifier: {drone_model_path}")

        self.worker_classes = self.worker_model.names
        self.ppe_classes = self.ppe_model.names

        logger.info(f"Worker classes: {self.worker_classes}")
        logger.info(f"PPE classes: {self.ppe_classes}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_crop(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Extract a worker crop with 10% margin. Returns (crop, crop_coords)."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        margin_x = (x2 - x1) * 0.1
        margin_y = (y2 - y1) * 0.1
        cx1 = max(0, int(x1 - margin_x))
        cy1 = max(0, int(y1 - margin_y))
        cx2 = min(w, int(x2 + margin_x))
        cy2 = min(h, int(y2 + margin_y))
        return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    def detect_workers(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
        """
        Detect workers in frame.

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            List of (bbox, confidence, class_id) for each detected worker
        """
        results = self.worker_model(
            frame,
            conf=self.conf_threshold,
            imgsz=self.input_size,
            half=self.fp16,
            verbose=False,
        )

        workers = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    workers.append(((x1, y1, x2, y2), conf, class_id))

        return workers

    def detect_ppe_batch(self, frame: np.ndarray,
                         workers: List[Tuple[Tuple[float, float, float, float], float, int]]
                         ) -> Dict[int, List[DetectionResult]]:
        """
        Detect PPE for all workers in a single batched YOLO call.

        Batching all crops into one inference call is significantly faster than
        calling the model once per worker when multiple workers are present.

        Args:
            frame: Full frame (H, W, 3) RGB
            workers: List of (bbox, confidence, class_id) from detect_workers()

        Returns:
            {worker_id: [DetectionResult]}
        """
        if not workers:
            return {}

        crops = []
        offsets = []  # (x_offset, y_offset) for each crop

        for bbox, _, _ in workers:
            crop, (cx1, cy1, cx2, cy2) = self._extract_crop(frame, bbox)
            if crop.size == 0:
                crops.append(None)
                offsets.append((0, 0))
            else:
                crops.append(crop)
                offsets.append((cx1, cy1))

        # Filter valid crops and run batch inference
        valid_indices = [i for i, c in enumerate(crops) if c is not None]
        valid_crops = [crops[i] for i in valid_indices]

        results_map: Dict[int, List[DetectionResult]] = {}

        if not valid_crops:
            return {i: [] for i in range(len(workers))}

        batch_results = self.ppe_model(
            valid_crops,
            conf=self.conf_threshold,
            imgsz=self.input_size,
            half=self.fp16,
            verbose=False,
        )

        for result_idx, worker_idx in enumerate(valid_indices):
            cx1, cy1 = offsets[worker_idx]
            ppe_items = []
            result = batch_results[result_idx]

            if result.boxes is not None:
                for box in result.boxes:
                    rx1, ry1, rx2, ry2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = self.ppe_classes[class_id]

                    # Remap crop-relative coords back to full-frame coords
                    x1_orig = cx1 + rx1
                    y1_orig = cy1 + ry1
                    x2_orig = cx1 + rx2
                    y2_orig = cy1 + ry2
                    area = int((x2_orig - x1_orig) * (y2_orig - y1_orig))

                    ppe_items.append(DetectionResult(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=conf,
                        bbox=(x1_orig, y1_orig, x2_orig, y2_orig),
                        area_pixels=area,
                    ))

            results_map[worker_idx] = ppe_items

        # Fill gaps (workers whose crop was empty)
        for i in range(len(workers)):
            if i not in results_map:
                results_map[i] = []

        return results_map

    def detect_ppe(self, frame: np.ndarray,
                   worker_bbox: Tuple[float, float, float, float]) -> List[DetectionResult]:
        """
        Detect PPE on a single worker crop.

        Prefer detect_ppe_batch() when multiple workers are present.
        """
        result = self.detect_ppe_batch(frame, [(worker_bbox, 1.0, 0)])
        return result.get(0, [])

    def process_frame(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """
        Process single frame: detect workers, then PPE on each worker.

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            {worker_id: [DetectionResult]} for PPE items on each worker
        """
        if self._use_roboflow:
            return self._roboflow.process_frame(frame)

        workers = self.detect_workers(frame)
        if not workers:
            return {}

        return self.detect_ppe_batch(frame, workers)

    def process_batch(self, frames: np.ndarray) -> List[Dict[int, List[DetectionResult]]]:
        """
        Process batch of frames.

        Args:
            frames: Batch of frames (B, H, W, 3) RGB

        Returns:
            List of {worker_id: [DetectionResult]} for each frame
        """
        return [self.process_frame(frame) for frame in frames]

    def warmup(self, frame_size: Tuple[int, int] = (640, 480), num_iterations: int = 3):
        """
        Warmup models to avoid startup latency.

        Args:
            frame_size: Frame size (height, width) for warmup
            num_iterations: Number of warmup iterations
        """
        if self._use_roboflow:
            self._roboflow.warmup(num_iterations=1)
            return

        logger.info("Warming up models...")

        for i in range(num_iterations):
            dummy = np.random.randint(0, 255, (frame_size[0], frame_size[1], 3), dtype=np.uint8)
            _ = self.worker_model(dummy, imgsz=self.input_size, half=self.fp16, verbose=False)
            _ = self.ppe_model(dummy, imgsz=self.input_size, half=self.fp16, verbose=False)
            logger.info(f"  Warmup iteration {i+1}/{num_iterations}")

        logger.info("✓ Models warmed up")

    def get_info(self) -> Dict:
        """Get pipeline information."""
        if self._use_roboflow:
            return self._roboflow.get_info()
        return {
            'device': str(self.device),
            'fp16': self.fp16,
            'input_size': self.input_size,
            'worker_model_classes': len(self.worker_classes),
            'ppe_model_classes': len(self.ppe_classes),
            'has_drone_classifier': self.drone_model is not None,
            'confidence_threshold': self.conf_threshold,
        }


class InferencePipelineWithMetrics(InferencePipeline):
    """Inference pipeline with built-in timing and metrics."""

    def __init__(self, *args, **kwargs):
        """Initialize with metrics tracking."""
        super().__init__(*args, **kwargs)
        self.timings = {
            'worker_detection': [],
            'ppe_detection': [],
            'total': []
        }

    def process_frame(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """Process frame with timing."""
        import time

        if self._use_roboflow:
            t0 = time.time()
            result = self._roboflow.process_frame(frame)
            elapsed = time.time() - t0
            self.timings['worker_detection'].append(elapsed)
            self.timings['ppe_detection'].append(0.0)
            self.timings['total'].append(elapsed)
            return result

        start_time = time.time()

        t0 = time.time()
        workers = self.detect_workers(frame)
        self.timings['worker_detection'].append(time.time() - t0)

        if not workers:
            self.timings['ppe_detection'].append(0.0)
            self.timings['total'].append(time.time() - start_time)
            return {}

        t0 = time.time()
        frame_detections = self.detect_ppe_batch(frame, workers)
        self.timings['ppe_detection'].append(time.time() - t0)
        self.timings['total'].append(time.time() - start_time)

        return frame_detections

    def get_metrics(self) -> Dict:
        """Get timing metrics."""
        if not self.timings['total']:
            return {}

        import numpy as np

        return {
            'worker_detection_ms': {
                'mean': np.mean(self.timings['worker_detection']) * 1000,
                'median': np.median(self.timings['worker_detection']) * 1000,
                'max': np.max(self.timings['worker_detection']) * 1000,
                'min': np.min(self.timings['worker_detection']) * 1000,
            },
            'ppe_detection_ms': {
                'mean': np.mean(self.timings['ppe_detection']) * 1000,
                'median': np.median(self.timings['ppe_detection']) * 1000,
                'max': np.max(self.timings['ppe_detection']) * 1000,
                'min': np.min(self.timings['ppe_detection']) * 1000,
            },
            'total_ms': {
                'mean': np.mean(self.timings['total']) * 1000,
                'median': np.median(self.timings['total']) * 1000,
                'max': np.max(self.timings['total']) * 1000,
                'min': np.min(self.timings['total']) * 1000,
            },
            'fps': 1.0 / np.mean(self.timings['total']) if self.timings['total'] else 0,
        }

    def reset_metrics(self):
        """Reset timing metrics."""
        self.timings = {
            'worker_detection': [],
            'ppe_detection': [],
            'total': []
        }
