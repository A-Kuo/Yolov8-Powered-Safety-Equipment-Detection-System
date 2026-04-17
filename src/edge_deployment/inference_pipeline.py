"""
Multi-model inference pipeline for PPE detection.

Phase 3: Orchestrate worker detection → optional drone filtering → PPE detection
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from ultralytics import YOLO
from src.edge_deployment.safety_rules_engine import DetectionResult


logger = logging.getLogger(__name__)


class InferencePipeline:
    """Coordinate multi-model inference: worker detection → PPE detection."""

    def __init__(self,
                 worker_model_path: str,
                 ppe_model_path: str,
                 drone_model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 conf_threshold: float = 0.25):
        """
        Initialize inference pipeline.

        Args:
            worker_model_path: Path to YOLOv8 worker detector model (.pt)
            ppe_model_path: Path to YOLOv8 PPE detector model (.pt)
            drone_model_path: Optional path to drone classifier
            device: Device to use ('cuda', 'cpu', etc.)
            conf_threshold: Global confidence threshold
        """
        self.device = device
        self.conf_threshold = conf_threshold

        # Load models
        logger.info(f"Loading models on {device}")

        self.worker_model = YOLO(worker_model_path)
        self.worker_model.to(device)
        logger.info(f"✓ Loaded worker detector: {worker_model_path}")

        self.ppe_model = YOLO(ppe_model_path)
        self.ppe_model.to(device)
        logger.info(f"✓ Loaded PPE detector: {ppe_model_path}")

        self.drone_model = None
        if drone_model_path and Path(drone_model_path).exists():
            self.drone_model = YOLO(drone_model_path)
            self.drone_model.to(device)
            logger.info(f"✓ Loaded drone classifier: {drone_model_path}")

        # Get class names
        self.worker_classes = self.worker_model.names
        self.ppe_classes = self.ppe_model.names

        logger.info(f"Worker classes: {self.worker_classes}")
        logger.info(f"PPE classes: {self.ppe_classes}")

    def detect_workers(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float, int]]:
        """
        Detect workers in frame.

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            List of (bbox, confidence, class_id) for each detected worker
        """
        # Run inference
        results = self.worker_model(frame, conf=self.conf_threshold, verbose=False)

        workers = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    workers.append(((x1, y1, x2, y2), conf, class_id))

        return workers

    def detect_ppe(self, frame: np.ndarray, worker_bbox: Tuple[float, float, float, float]) -> List[DetectionResult]:
        """
        Detect PPE items on a worker.

        Args:
            frame: Input frame (H, W, 3) RGB
            worker_bbox: Worker bounding box (x1, y1, x2, y2)

        Returns:
            List of DetectionResult for PPE items
        """
        x1, y1, x2, y2 = worker_bbox

        # Expand crop slightly for context
        h, w = frame.shape[:2]
        margin_x = (x2 - x1) * 0.1
        margin_y = (y2 - y1) * 0.1

        x1_crop = max(0, int(x1 - margin_x))
        y1_crop = max(0, int(y1 - margin_y))
        x2_crop = min(w, int(x2 + margin_x))
        y2_crop = min(h, int(y2 + margin_y))

        # Extract crop
        crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        if crop.size == 0:
            return []

        # Run PPE detection on crop
        results = self.ppe_model(crop, conf=self.conf_threshold, verbose=False)

        ppe_items = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1_crop_rel, y1_crop_rel, x2_crop_rel, y2_crop_rel = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = self.ppe_classes[class_id]

                    # Convert back to original frame coordinates
                    x1_orig = x1_crop + x1_crop_rel
                    y1_orig = y1_crop + y1_crop_rel
                    x2_orig = x1_crop + x2_crop_rel
                    y2_orig = y1_crop + y2_crop_rel

                    # Calculate area
                    area = (x2_orig - x1_orig) * (y2_orig - y1_orig)

                    ppe_items.append(DetectionResult(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=conf,
                        bbox=(x1_orig, y1_orig, x2_orig, y2_orig),
                        area_pixels=int(area)
                    ))

        return ppe_items

    def process_frame(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """
        Process single frame: detect workers, then PPE on each worker.

        Args:
            frame: Input frame (H, W, 3) RGB

        Returns:
            {worker_id: [DetectionResult]} for PPE items on each worker
        """
        frame_detections = {}

        # Step 1: Detect workers
        workers = self.detect_workers(frame)

        if not workers:
            return frame_detections

        # Step 2: For each worker, detect PPE
        for worker_id, (bbox, worker_conf, worker_class_id) in enumerate(workers):
            ppe_items = self.detect_ppe(frame, bbox)
            frame_detections[worker_id] = ppe_items

        return frame_detections

    def process_batch(self, frames: np.ndarray) -> List[Dict[int, List[DetectionResult]]]:
        """
        Process batch of frames.

        Args:
            frames: Batch of frames (B, H, W, 3) RGB

        Returns:
            List of {worker_id: [DetectionResult]} for each frame
        """
        batch_results = []

        for frame in frames:
            result = self.process_frame(frame)
            batch_results.append(result)

        return batch_results

    def warmup(self, frame_size: Tuple[int, int] = (640, 480), num_iterations: int = 3):
        """
        Warmup models to avoid startup latency.

        Args:
            frame_size: Frame size (height, width) for warmup
            num_iterations: Number of warmup iterations
        """
        logger.info("Warming up models...")

        for i in range(num_iterations):
            dummy_frame = np.random.randint(0, 255, (frame_size[0], frame_size[1], 3), dtype=np.uint8)

            # Warmup worker detector
            _ = self.worker_model(dummy_frame, conf=self.conf_threshold, verbose=False)

            # Warmup PPE detector
            _ = self.ppe_model(dummy_frame, conf=self.conf_threshold, verbose=False)

            logger.info(f"  Warmup iteration {i+1}/{num_iterations}")

        logger.info("✓ Models warmed up")

    def get_info(self) -> Dict:
        """Get pipeline information."""
        return {
            'device': str(self.device),
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

        start_time = time.time()

        frame_detections = {}

        # Step 1: Detect workers
        t0 = time.time()
        workers = self.detect_workers(frame)
        worker_time = time.time() - t0
        self.timings['worker_detection'].append(worker_time)

        if not workers:
            return frame_detections

        # Step 2: For each worker, detect PPE
        t0 = time.time()
        for worker_id, (bbox, worker_conf, worker_class_id) in enumerate(workers):
            ppe_items = self.detect_ppe(frame, bbox)
            frame_detections[worker_id] = ppe_items
        ppe_time = time.time() - t0
        self.timings['ppe_detection'].append(ppe_time)

        total_time = time.time() - start_time
        self.timings['total'].append(total_time)

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
