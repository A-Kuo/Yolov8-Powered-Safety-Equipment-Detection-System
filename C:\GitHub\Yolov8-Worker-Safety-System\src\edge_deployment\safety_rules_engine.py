"""Safety Compliance Rules Engine

Policy-driven compliance checking with:
- YAML configurable rules
- Temporal smoothing
- Alert deduplication
- Per-worker scoring
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Safety alert."""
    severity: str  # 'critical', 'warning', 'info'
    equipment: str
    message: str


@dataclass
class ComplianceResult:
    """Compliance evaluation result."""
    worker_id: int
    is_compliant: bool
    confidence_score: float
    missing_equipment: List[str]
    alerts: List[Alert]


class SafetyRulesEngine:
    """Policy-driven compliance engine."""

    def __init__(
        self,
        config_path: str,
        temporal_window: int = 0,
        alert_cooldown_frames: int = 30,
    ):
        """Initialize engine.

        Args:
            config_path: Path to compliance_rules.yaml
            temporal_window: Frames for smoothing (0 = disabled)
            alert_cooldown_frames: Suppress repeat alerts for N frames
        """
        self.config_path = Path(config_path)
        self.temporal_window = temporal_window
        self.alert_cooldown_frames = alert_cooldown_frames

        # Load config
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config not found: {config_path}")
            self.config = {}

        # State tracking
        self._detection_history = defaultdict(lambda: deque(maxlen=temporal_window or 1))
        self._alert_last_frame = defaultdict(int)

        logger.info(f"Initialized SafetyRulesEngine (window={temporal_window})")

    def evaluate_worker(
        self,
        worker_id: int,
        ppe_items: List[Any],
        frame_idx: int,
    ) -> ComplianceResult:
        """Evaluate compliance for single worker.

        Args:
            worker_id: Worker identifier
            ppe_items: List of detected PPE items (DetectionResult objects)
            frame_idx: Current frame number

        Returns:
            ComplianceResult with compliance verdict
        """
        # Extract equipment
        detected = {item.class_name for item in ppe_items}
        self._detection_history[worker_id].append(detected)

        # Required equipment
        required = set(self.config.get('required_equipment', []))

        # Check compliance
        missing = required - detected
        is_compliant = len(missing) == 0

        # Calculate confidence
        confidence = 1.0 - (len(missing) / max(len(required), 1))

        # Generate alerts
        alerts = []
        for equipment in missing:
            alert = self._should_fire_alert(worker_id, equipment, frame_idx)
            if alert:
                alerts.append(alert)

        return ComplianceResult(
            worker_id=worker_id,
            is_compliant=is_compliant,
            confidence_score=confidence,
            missing_equipment=list(missing),
            alerts=alerts,
        )

    def _should_fire_alert(
        self,
        worker_id: int,
        equipment: str,
        frame_idx: int,
    ) -> Optional[Alert]:
        """Check if alert should fire (cooldown suppression)."""
        key = (worker_id, equipment)
        last_frame = self._alert_last_frame.get(key, -self.alert_cooldown_frames)

        if frame_idx - last_frame >= self.alert_cooldown_frames:
            self._alert_last_frame[key] = frame_idx
            severity = self.config.get('equipment', {}).get(equipment, {}).get('severity', 'warning')
            return Alert(
                severity=severity,
                equipment=equipment,
                message=f"Worker {worker_id}: Missing {equipment}",
            )

        return None
