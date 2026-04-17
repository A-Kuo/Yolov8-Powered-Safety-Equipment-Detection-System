"""
Metrics collection and analysis for inference results.

Phase 3: Track detection statistics, confidence distributions, compliance metrics.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for single PPE class."""
    class_name: str
    total_detections: int
    mean_confidence: float
    std_confidence: float
    min_confidence: float
    max_confidence: float
    detection_rate: float  # % of frames this class appears


@dataclass
class FrameMetrics:
    """Metrics for single frame."""
    frame_idx: int
    total_workers: int
    total_ppe_items: int
    compliant_workers: int
    non_compliant_workers: int
    compliance_rate: float


@dataclass
class VideoMetrics:
    """Aggregated metrics for entire video."""
    total_frames: int
    total_workers: int
    total_ppe_detections: int
    avg_workers_per_frame: float
    avg_ppe_per_worker: float
    avg_compliance_rate: float
    class_metrics: Dict[str, ClassMetrics]
    frame_metrics: List[FrameMetrics]


class MetricsCollector:
    """Collect and analyze inference metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.frame_metrics = []
        self.class_detections = defaultdict(list)  # {class_name: [confidences]}
        self.class_frequencies = defaultdict(int)  # {class_name: count}
        self.total_frames = 0
        self.total_workers = 0
        self.total_compliant = 0

    def record_frame(self,
                    frame_idx: int,
                    frame_detections: Dict,
                    compliance_results: Optional[Dict] = None):
        """
        Record metrics for single frame.

        Args:
            frame_idx: Frame index
            frame_detections: {worker_id: [DetectionResult]}
            compliance_results: {worker_id: ComplianceResult} (optional)
        """
        total_workers = len(frame_detections)
        total_ppe_items = sum(len(items) for items in frame_detections.values())

        # Track class detections
        for worker_id, ppe_items in frame_detections.items():
            for item in ppe_items:
                self.class_detections[item.class_name].append(item.confidence)
                self.class_frequencies[item.class_name] += 1

        # Compliance metrics
        compliant_workers = 0
        if compliance_results:
            compliant_workers = sum(1 for r in compliance_results.values() if r.is_compliant)

        self.total_workers += total_workers
        self.total_compliant += compliant_workers
        self.total_frames += 1

        # Frame-level metrics
        compliance_rate = compliant_workers / total_workers if total_workers > 0 else 0

        frame_metric = FrameMetrics(
            frame_idx=frame_idx,
            total_workers=total_workers,
            total_ppe_items=total_ppe_items,
            compliant_workers=compliant_workers,
            non_compliant_workers=total_workers - compliant_workers,
            compliance_rate=compliance_rate
        )

        self.frame_metrics.append(frame_metric)

    def get_class_metrics(self) -> Dict[str, ClassMetrics]:
        """Get per-class metrics."""
        metrics = {}

        for class_name, confidences in self.class_detections.items():
            if not confidences:
                continue

            confidences = np.array(confidences)

            # Frequency as % of frames
            detection_rate = self.class_frequencies[class_name] / self.total_frames if self.total_frames > 0 else 0

            metrics[class_name] = ClassMetrics(
                class_name=class_name,
                total_detections=len(confidences),
                mean_confidence=float(np.mean(confidences)),
                std_confidence=float(np.std(confidences)),
                min_confidence=float(np.min(confidences)),
                max_confidence=float(np.max(confidences)),
                detection_rate=detection_rate
            )

        return metrics

    def get_summary(self) -> VideoMetrics:
        """Get aggregated video metrics."""
        class_metrics = self.get_class_metrics()

        avg_workers_per_frame = self.total_workers / self.total_frames if self.total_frames > 0 else 0
        total_ppe = sum(len(items) for _, items in self.class_detections.items())
        avg_ppe_per_worker = total_ppe / self.total_workers if self.total_workers > 0 else 0
        avg_compliance_rate = self.total_compliant / self.total_workers if self.total_workers > 0 else 0

        return VideoMetrics(
            total_frames=self.total_frames,
            total_workers=self.total_workers,
            total_ppe_detections=total_ppe,
            avg_workers_per_frame=avg_workers_per_frame,
            avg_ppe_per_worker=avg_ppe_per_worker,
            avg_compliance_rate=avg_compliance_rate,
            class_metrics=class_metrics,
            frame_metrics=self.frame_metrics
        )

    def print_summary(self):
        """Print metrics summary."""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("DETECTION METRICS SUMMARY")
        print("="*70)

        print(f"\nFrames Analyzed:       {summary.total_frames}")
        print(f"Total Workers:         {summary.total_workers}")
        print(f"Total PPE Detections:  {summary.total_ppe_detections}")

        print(f"\nAverages:")
        print(f"  Workers per frame:   {summary.avg_workers_per_frame:.1f}")
        print(f"  PPE items per worker: {summary.avg_ppe_per_worker:.1f}")
        print(f"  Compliance rate:     {summary.avg_compliance_rate:.1%}")

        print(f"\nClass Detection Rates:")
        for class_name in sorted(summary.class_metrics.keys()):
            metrics = summary.class_metrics[class_name]
            print(f"  {class_name:25s}: {metrics.detection_rate:5.1%} | "
                  f"Conf: {metrics.mean_confidence:.2f} ± {metrics.std_confidence:.2f}")

        print("\n" + "="*70 + "\n")

        return summary

    def get_confidence_distribution(self, class_name: str) -> Dict:
        """Get confidence distribution for class."""
        confidences = self.class_detections.get(class_name, [])
        if not confidences:
            return {}

        confidences = np.array(confidences)

        return {
            'count': len(confidences),
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'median': float(np.median(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'p25': float(np.percentile(confidences, 25)),
            'p75': float(np.percentile(confidences, 75)),
            'p95': float(np.percentile(confidences, 95)),
        }

    def get_false_positive_analysis(self, ground_truth_classes: Optional[List[str]] = None) -> Dict:
        """
        Analyze false positives (optional, requires ground truth).

        Args:
            ground_truth_classes: Ground truth PPE classes in video

        Returns:
            False positive analysis
        """
        detected_classes = set(self.class_detections.keys())

        if ground_truth_classes is None:
            logger.warning("Ground truth not provided, cannot analyze false positives")
            return {}

        ground_truth_set = set(ground_truth_classes)

        false_positives = detected_classes - ground_truth_set
        missed_detections = ground_truth_set - detected_classes
        correct_detections = detected_classes & ground_truth_set

        return {
            'false_positives': list(false_positives),
            'missed_detections': list(missed_detections),
            'correct_detections': list(correct_detections),
        }

    def export_json(self, output_path: str):
        """Export metrics to JSON."""
        import json

        summary = self.get_summary()

        data = {
            'summary': {
                'total_frames': summary.total_frames,
                'total_workers': summary.total_workers,
                'total_ppe_detections': summary.total_ppe_detections,
                'avg_workers_per_frame': summary.avg_workers_per_frame,
                'avg_ppe_per_worker': summary.avg_ppe_per_worker,
                'avg_compliance_rate': summary.avg_compliance_rate,
            },
            'class_metrics': {
                name: asdict(metrics)
                for name, metrics in summary.class_metrics.items()
            },
            'frame_metrics': [asdict(fm) for fm in summary.frame_metrics]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported metrics to {output_path}")

    def export_csv(self, output_path: str):
        """Export frame metrics to CSV."""
        import csv

        summary = self.get_summary()

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.frame_metrics[0]).keys())
            writer.writeheader()
            for fm in summary.frame_metrics:
                writer.writerow(asdict(fm))

        logger.info(f"Exported frame metrics to {output_path}")

    def plot_confidence_distribution(self, class_name: str, output_path: Optional[str] = None):
        """Plot confidence distribution for class."""
        try:
            import matplotlib.pyplot as plt

            confidences = self.class_detections.get(class_name, [])
            if not confidences:
                logger.warning(f"No detections for {class_name}")
                return

            plt.figure(figsize=(10, 6))
            plt.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
            plt.axvline(x=np.mean(confidences), color='r', linestyle='--', label=f'Mean: {np.mean(confidences):.2f}')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            plt.title(f'Confidence Distribution: {class_name}')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

            if output_path:
                plt.savefig(output_path, dpi=100)
                logger.info(f"Saved plot to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")

    def plot_class_detection_rates(self, output_path: Optional[str] = None):
        """Plot detection rates for all classes."""
        try:
            import matplotlib.pyplot as plt

            summary = self.get_summary()
            class_names = sorted(summary.class_metrics.keys())
            detection_rates = [summary.class_metrics[cn].detection_rate for cn in class_names]

            plt.figure(figsize=(12, 6))
            plt.bar(class_names, detection_rates)
            plt.xlabel('PPE Class')
            plt.ylabel('Detection Rate')
            plt.title('PPE Detection Rates Across Video')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=100)
                logger.info(f"Saved plot to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")

    def plot_compliance_rate_over_time(self, output_path: Optional[str] = None):
        """Plot compliance rate over time."""
        try:
            import matplotlib.pyplot as plt

            frames = [fm.frame_idx for fm in self.frame_metrics]
            compliance_rates = [fm.compliance_rate for fm in self.frame_metrics]

            plt.figure(figsize=(12, 6))
            plt.plot(frames, compliance_rates, 'b-', alpha=0.7)
            plt.axhline(y=np.mean(compliance_rates), color='r', linestyle='--',
                       label=f'Average: {np.mean(compliance_rates):.1%}')
            plt.xlabel('Frame Index')
            plt.ylabel('Compliance Rate')
            plt.title('Worker Compliance Rate Over Time')
            plt.ylim([0, 1])
            plt.legend()
            plt.grid(True, alpha=0.3)

            if output_path:
                plt.savefig(output_path, dpi=100)
                logger.info(f"Saved plot to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
