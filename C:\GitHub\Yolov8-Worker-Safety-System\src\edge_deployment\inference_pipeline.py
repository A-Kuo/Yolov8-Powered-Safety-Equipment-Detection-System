"""Inference Pipeline with Metrics

Orchestrates:
- Worker detection (YOLOv8-N)
- PPE detection (YOLOv8-M)
- Roboflow cloud backend (optional)
- Performance metrics collection
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class InferencePipelineWithMetrics:
    """Dual-model inference pipeline with metrics."""

    def __init__(
        self,
        worker_model_path: str,
        ppe_model_path: str,
        device: str = 'cuda',
        use_roboflow: bool = False,
        fp16: bool = False,
        input_size: int = 640,
    ):
        """Initialize pipeline.

        Args:
            worker_model_path: Path to worker detector (.pt)
            ppe_model_path: Path to PPE detector (.pt)
            device: 'cuda' or 'cpu'
            use_roboflow: Use Roboflow cloud backend if True
            fp16: Use half precision
            input_size: Model input resolution
        """
        self.device = device
        self.use_roboflow = use_roboflow
        self.fp16 = fp16
        self.input_size = input_size

        # Load models
        if not use_roboflow:
            self.worker_model = YOLO(worker_model_path)
            self.ppe_model = YOLO(ppe_model_path)
            logger.info(f"Loaded local models on {device}")
        else:
            self.worker_model = None
            self.ppe_model = None
            try:
                from inference_sdk import InferenceHTTPClient
                import os
                api_key = os.environ.get('ROBOFLOW_API_KEY')
                if not api_key:
                    raise ValueError("ROBOFLOW_API_KEY not set")
                self.roboflow_client = InferenceHTTPClient(
                    api_url='https://detect.roboflow.com',
                    api_key=api_key,
                )
                logger.info("Initialized Roboflow cloud backend")
            except Exception as e:
                raise RuntimeError(f"Roboflow init failed: {e}")

        # Metrics
        self.metrics = {
            'frames_processed': 0,
            'latencies': [],
            'worker_counts': [],
        }

    def process_frame(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """Process single frame.

        Returns:
            {worker_id: [DetectionResult, ...], ...}
        """
        t0 = time.perf_counter()

        if self.use_roboflow:
            detections = self._roboflow_inference(frame)
        else:
            detections = self._local_inference(frame)

        latency_ms = (time.perf_counter() - t0) * 1000
        self.metrics['latencies'].append(latency_ms)
        self.metrics['frames_processed'] += 1

        return detections

    def _local_inference(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """Local YOLOv8 inference."""
        # Worker detection
        worker_results = self.worker_model(
            frame,
            conf=0.5,
            imgsz=self.input_size,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )

        workers = {}
        if len(worker_results) > 0 and worker_results[0].boxes is not None:
            for box in worker_results[0].boxes:
                wid = int(box.id) if box.id is not None else 0
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                workers[wid] = [int(x1), int(y1), int(x2), int(y2)]

        # PPE detection
        ppe_results = self.ppe_model(
            frame,
            conf=0.4,
            imgsz=self.input_size,
            device=self.device,
            half=self.fp16,
            verbose=False,
        )

        ppe_dict = {}
        if len(ppe_results) > 0 and ppe_results[0].boxes is not None:
            for box in ppe_results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                class_name = ppe_results[0].names.get(class_id, f"class_{class_id}")
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = [float(x1), float(y1), float(x2), float(y2)]

                # Assign to nearest worker or default
                wid = 0
                if workers:
                    best_dist = float('inf')
                    for w_id, (wx1, wy1, wx2, wy2) in workers.items():
                        w_cx, w_cy = (wx1 + wx2) / 2, (wy1 + wy2) / 2
                        p_cx, p_cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = ((p_cx - w_cx) ** 2 + (p_cy - w_cy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            wid = w_id

                if wid not in ppe_dict:
                    ppe_dict[wid] = []
                ppe_dict[wid].append(
                    DetectionResult(class_name, class_id, conf, bbox)
                )

        self.metrics['worker_counts'].append(len(workers))
        return ppe_dict

    def _roboflow_inference(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """Roboflow cloud inference."""
        try:
            # Encode frame
            import cv2
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = buffer.tobytes()

            # Call Roboflow
            result = self.roboflow_client.infer_from_image(
                frame_base64,
                model_id='construction-safety-0/1',
            )

            # Parse results
            ppe_dict = {}
            if 'predictions' in result:
                for pred in result['predictions']:
                    class_name = pred.get('class', 'unknown')
                    conf = pred.get('confidence', 0.0)
                    x = pred.get('x', 0)
                    y = pred.get('y', 0)
                    w = pred.get('width', 0)
                    h = pred.get('height', 0)
                    bbox = [x - w/2, y - h/2, x + w/2, y + h/2]

                    wid = 0
                    if wid not in ppe_dict:
                        ppe_dict[wid] = []
                    ppe_dict[wid].append(
                        DetectionResult(class_name, 0, conf, bbox)
                    )

            return ppe_dict
        except Exception as e:
            logger.error(f"Roboflow inference failed: {e}")
            return {}

    def detect_ppe_batch(
        self,
        frame: np.ndarray,
        workers: Dict[int, tuple],
    ) -> Dict[int, List[DetectionResult]]:
        """Batched PPE detection on worker regions."""
        if not workers or self.use_roboflow:
            return self.process_frame(frame)

        # Extract crops
        crops = {}
        for wid, (x1, y1, x2, y2) in workers.items():
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                        max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                crops[wid] = crop

        if not crops:
            return {}

        # Batch inference
        ppe_dict = {}
        for wid, crop in crops.items():
            results = self.ppe_model(
                crop,
                conf=0.4,
                imgsz=self.input_size,
                device=self.device,
                half=self.fp16,
                verbose=False,
            )

            if len(results) > 0 and results[0].boxes is not None:
                ppe_dict[wid] = []
                for box in results[0].boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = results[0].names.get(class_id, f"class_{class_id}")
                    conf = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    ppe_dict[wid].append(
                        DetectionResult(class_name, class_id, conf, bbox)
                    )

        return ppe_dict

    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics."""
        lats = self.metrics['latencies']
        return {
            'frames_processed': self.metrics['frames_processed'],
            'fps': len(lats) / (sum(lats) / 1000) if lats else 0,
            'total_ms': {
                'mean': np.mean(lats) if lats else 0,
                'p95': np.percentile(lats, 95) if lats else 0,
            },
        }
