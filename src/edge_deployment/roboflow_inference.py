"""
Roboflow Workflow inference for PPE safety detection.

Uses the pre-built PPE Compliance Pipeline (Workflow ID: zGLpQAKajlvk32DknfR6)
which handles worker tracking (rfdetr-nano) and PPE detection (ppe-hgqzw/6)
with 8,700+ real-world training images — no local model required.

Usage:
    client = RoboflowInference(api_key=os.getenv('ROBOFLOW_API_KEY'))
    results = client.process_frame(frame)
"""

import os
import io
import base64
import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from src.edge_deployment.safety_rules_engine import DetectionResult


logger = logging.getLogger(__name__)

# Workflow constants — update if workflow is redeployed
_WORKSPACE  = "austins-workspace-gjnf8"
_WORKFLOW   = "zGLpQAKajlvk32DknfR6"
_API_URL    = "https://detect.roboflow.com"

# Map Roboflow class labels → project 10-class schema
# Adjust if the workflow returns different label strings
_CLASS_MAP: Dict[str, str] = {
    # Worker / person
    "person":           "worker",
    "worker":           "worker",
    # Head
    "hard-hat":         "hard_hat",
    "hardhat":          "hard_hat",
    "hard_hat":         "hard_hat",
    "helmet":           "hard_hat",
    "no-hardhat":       "regular_hat",
    "no_hardhat":       "regular_hat",
    # Eye
    "safety-glasses":   "safety_glasses",
    "safety_glasses":   "safety_glasses",
    "glasses":          "safety_glasses",
    "goggles":          "safety_goggles",
    "safety_goggles":   "safety_goggles",
    # Torso
    "safety-vest":      "hi_vis_vest",
    "safety_vest":      "hi_vis_vest",
    "vest":             "hi_vis_vest",
    "no-safety-vest":   "regular_clothing",
    "no_safety_vest":   "regular_clothing",
    # Foot
    "boots":            "work_boots",
    "safety-boots":     "work_boots",
    "safety_boots":     "work_boots",
    "work_boots":       "work_boots",
    "shoes":            "regular_shoes",
    # Drone
    "drone":            "drone",
}


def _encode_frame(frame: np.ndarray) -> str:
    """Encode numpy RGB frame to base64 JPEG string for API upload."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _normalise_class(raw: str) -> str:
    """Map a raw Roboflow class label to the project schema, or pass through."""
    return _CLASS_MAP.get(raw.lower().replace(" ", "_"),
                          _CLASS_MAP.get(raw.lower().replace(" ", "-"), raw.lower()))


def _parse_predictions(raw_predictions) -> List[DetectionResult]:
    """
    Convert Roboflow prediction dicts to DetectionResult list.

    Roboflow returns predictions in two common shapes:
      1. {'x', 'y', 'width', 'height', 'class', 'confidence', ...}
      2. Nested inside 'predictions' key
    """
    results: List[DetectionResult] = []

    items: list = []
    if isinstance(raw_predictions, list):
        items = raw_predictions
    elif isinstance(raw_predictions, dict):
        items = raw_predictions.get("predictions", [])

    for pred in items:
        try:
            raw_class = pred.get("class", pred.get("class_name", "unknown"))
            class_name = _normalise_class(raw_class)
            confidence = float(pred.get("confidence", 0.0))

            # Roboflow uses centre-format (x, y, width, height)
            cx = float(pred.get("x", 0))
            cy = float(pred.get("y", 0))
            w  = float(pred.get("width", 0))
            h  = float(pred.get("height", 0))

            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            area = int(w * h)

            # Assign a numeric class_id from schema (0-9) or -1 if unknown
            schema = [
                "worker", "drone", "safety_glasses", "safety_goggles",
                "hard_hat", "regular_hat", "hi_vis_vest", "regular_clothing",
                "work_boots", "regular_shoes",
            ]
            class_id = schema.index(class_name) if class_name in schema else -1

            results.append(DetectionResult(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                area_pixels=area,
            ))

        except Exception as exc:
            logger.debug(f"Skipping malformed prediction: {pred} — {exc}")

    return results


class RoboflowInference:
    """
    Wraps the Roboflow PPE Compliance Pipeline workflow.

    The workflow executes two models server-side:
      • rfdetr-nano  — worker / person tracking
      • ppe-hgqzw/6 — fine-grained PPE detection

    One call to `process_frame` replaces the entire local two-model pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: str = _WORKSPACE,
        workflow_id: str = _WORKFLOW,
        api_url: str = _API_URL,
        conf_threshold: float = 0.25,
    ):
        """
        Args:
            api_key:      Roboflow API key. Falls back to ROBOFLOW_API_KEY env var.
            workspace:    Roboflow workspace slug.
            workflow_id:  Workflow ID to invoke.
            api_url:      Roboflow inference server URL.
            conf_threshold: Minimum confidence to keep a detection.
        """
        self.api_key       = api_key or os.getenv("ROBOFLOW_API_KEY", "")
        self.workspace     = workspace
        self.workflow_id   = workflow_id
        self.api_url       = api_url
        self.conf_threshold = conf_threshold

        if not self.api_key:
            raise ValueError(
                "Roboflow API key is required. "
                "Set ROBOFLOW_API_KEY env var or pass api_key=..."
            )

        # Lazy-import so the package is optional when using local models
        try:
            from inference_sdk import InferenceHTTPClient
            self._client = InferenceHTTPClient(
                api_url=self.api_url,
                api_key=self.api_key,
            )
            logger.info(f"✓ Roboflow client ready — workflow: {self.workflow_id}")
        except ImportError as exc:
            raise ImportError(
                "inference_sdk is not installed. "
                "Run: pip install inference-sdk"
            ) from exc

        # Timing
        self.timings: Dict[str, List[float]] = {"total": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Dict[int, List[DetectionResult]]:
        """
        Run the Roboflow workflow on a single frame.

        Args:
            frame: RGB numpy array (H, W, 3).

        Returns:
            {worker_id: [DetectionResult]}  — same contract as InferencePipeline.
        """
        t0 = time.time()

        try:
            encoded = _encode_frame(frame)
            raw = self._client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow_id,
                images={"image": encoded},
            )
        except Exception as exc:
            logger.error(f"Roboflow workflow call failed: {exc}")
            return {}

        self.timings["total"].append(time.time() - t0)
        return self._parse_workflow_result(raw)

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        target_fps: int = 30,
    ) -> List[Dict[int, List[DetectionResult]]]:
        """
        Run workflow on every frame of a video file.

        Args:
            video_path:  Path to MP4 / AVI / MOV.
            max_frames:  Cap number of frames (useful for testing).
            target_fps:  Desired frame rate; skips frames to match.

        Returns:
            List of per-frame {worker_id: [DetectionResult]}.
        """
        from src.edge_deployment.video_processor import VideoProcessor

        results = []
        with VideoProcessor(video_path, target_fps=target_fps) as vp:
            for frame, idx, _ in vp:
                if max_frames and idx >= max_frames:
                    break
                results.append(self.process_frame(frame))

        return results

    def get_metrics(self) -> Dict:
        """Return latency statistics (matches InferencePipelineWithMetrics API)."""
        totals = self.timings["total"]
        if not totals:
            return {}
        arr = np.array(totals) * 1000  # → ms
        return {
            "total_ms": {
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "max":    float(np.max(arr)),
                "min":    float(np.min(arr)),
            },
            "fps": 1_000 / float(np.mean(arr)) if arr.size else 0,
            "frames_processed": len(totals),
        }

    def get_info(self) -> Dict:
        """Return configuration info."""
        return {
            "backend":     "roboflow_workflow",
            "workspace":   self.workspace,
            "workflow_id": self.workflow_id,
            "api_url":     self.api_url,
            "conf_threshold": self.conf_threshold,
        }

    def warmup(self, num_iterations: int = 1):
        """Send a small dummy image to pre-warm the API connection."""
        logger.info("Warming up Roboflow connection…")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(num_iterations):
            self.process_frame(dummy)
        logger.info("✓ Roboflow warmup done")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_workflow_result(
        self, raw
    ) -> Dict[int, List[DetectionResult]]:
        """
        Convert the raw workflow response into {worker_id: [DetectionResult]}.

        The Roboflow workflow returns a list with one element per input image.
        Each element contains the combined predictions from all pipeline stages.
        """
        frame_detections: Dict[int, List[DetectionResult]] = {}

        if not raw:
            return frame_detections

        # raw is typically: [{"output_1": {...}, "output_2": {...}}]
        first = raw[0] if isinstance(raw, list) else raw

        # Collect all prediction lists across all workflow output keys
        all_preds: list = []
        if isinstance(first, dict):
            for val in first.values():
                if isinstance(val, dict):
                    preds = val.get("predictions", [])
                    if isinstance(preds, list):
                        all_preds.extend(preds)
                elif isinstance(val, list):
                    all_preds.extend(val)

        if not all_preds:
            # Fallback: try the top-level list directly
            if isinstance(first, list):
                all_preds = first

        detections = _parse_predictions(all_preds)
        detections = [d for d in detections if d.confidence >= self.conf_threshold]

        if not detections:
            return frame_detections

        # Separate workers from PPE items
        worker_detections = [d for d in detections if d.class_name == "worker"]
        ppe_detections    = [d for d in detections if d.class_name != "worker"]

        if not worker_detections:
            # No explicit worker boxes — treat all PPE as belonging to worker 0
            frame_detections[0] = ppe_detections
            return frame_detections

        # Assign each PPE item to the closest worker by IoU / centre proximity
        for w_id, worker in enumerate(worker_detections):
            frame_detections[w_id] = _assign_ppe_to_worker(
                worker.bbox, ppe_detections
            )

        return frame_detections


# ---------------------------------------------------------------------------
# Utility: assign PPE boxes to a worker box
# ---------------------------------------------------------------------------

def _assign_ppe_to_worker(
    worker_bbox: Tuple[float, float, float, float],
    ppe_items: List[DetectionResult],
    iou_threshold: float = 0.0,   # any overlap counts
) -> List[DetectionResult]:
    """
    Return the subset of ppe_items whose centre falls inside worker_bbox,
    or that overlap with worker_bbox (for partially visible equipment).
    """
    wx1, wy1, wx2, wy2 = worker_bbox
    assigned = []

    for item in ppe_items:
        ix1, iy1, ix2, iy2 = item.bbox
        # Item centre
        cx, cy = (ix1 + ix2) / 2, (iy1 + iy2) / 2
        if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
            assigned.append(item)
            continue
        # Overlap check as fallback
        overlap_x = max(0, min(wx2, ix2) - max(wx1, ix1))
        overlap_y = max(0, min(wy2, iy2) - max(wy1, iy1))
        if overlap_x > 0 and overlap_y > 0:
            assigned.append(item)

    return assigned
