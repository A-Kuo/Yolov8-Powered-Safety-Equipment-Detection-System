#!/usr/bin/env python3
"""
End-to-end local video inference with compliance checking and performance profiling.

Phase 3: Complete pipeline for testing models on local warehouse videos.

Usage:
    python scripts/run_local_video_inference.py \
        --video /path/to/video.mp4 \
        --worker-model models/yolo/ppe_detector.pt \
        --ppe-model models/yolo/ppe_detector.pt \
        --output output/ \
        --fps 30

Output:
    - output/annotated_video.mp4 (with detections and compliance status)
    - output/results.json (frame-by-frame detections)
    - output/compliance_report.json (worker compliance analysis)
    - output/performance.json (FPS, latency, memory metrics)
    - output/plots/ (various performance and detection plots)
"""

import argparse
import logging
import json
import sys
from pathlib import Path

import numpy as np
import cv2

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.edge_deployment.video_processor import VideoProcessor, VideoWriter
from src.edge_deployment.inference_pipeline import InferencePipelineWithMetrics
from src.edge_deployment.safety_rules_engine import SafetyRulesEngine, AlertSeverity
from src.edge_deployment.performance_profiler import PerformanceProfiler
from src.edge_deployment.metrics_collector import MetricsCollector


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def draw_detections(frame: np.ndarray, detections: dict, compliance_results: dict) -> np.ndarray:
    """
    Draw detection boxes and compliance status on frame.

    Args:
        frame: Input frame (RGB)
        detections: {worker_id: [DetectionResult]}
        compliance_results: {worker_id: ComplianceResult}

    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Color palette
    colors = {
        'compliant': (0, 255, 0),      # Green
        'non_compliant': (0, 0, 255),  # Red
        'uncertain': (255, 165, 0),    # Orange
    }

    for worker_id, ppe_items in detections.items():
        # Get compliance status
        compliance = compliance_results.get(worker_id)
        is_compliant = compliance.is_compliant if compliance else None

        if is_compliant is True:
            status_color = colors['compliant']
            status_text = "✓ COMPLIANT"
        elif is_compliant is False:
            status_color = colors['non_compliant']
            status_text = "✗ NON-COMPLIANT"
        else:
            status_color = colors['uncertain']
            status_text = "? UNCERTAIN"

        # Draw PPE bounding boxes
        for item in ppe_items:
            x1, y1, x2, y2 = item.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), status_color, 2)

            # Draw label
            label = f"{item.class_name} {item.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, status_color, 1
            )

    return annotated


def process_video(video_path: str,
                 worker_model_path: str,
                 ppe_model_path: str,
                 compliance_config_path: str,
                 output_dir: str,
                 target_fps: int = 30,
                 max_frames: int = None) -> dict:
    """
    Process entire video with inference and compliance checking.

    Args:
        video_path: Path to input video
        worker_model_path: Path to worker detection model
        ppe_model_path: Path to PPE detection model
        compliance_config_path: Path to compliance rules YAML
        output_dir: Output directory
        target_fps: Resample video to this FPS
        max_frames: Limit number of frames (for testing)

    Returns:
        Results dictionary with all metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing video: {video_path}")

    # Initialize components
    video_processor = VideoProcessor(video_path, target_fps=target_fps)
    pipeline = InferencePipelineWithMetrics(
        worker_model_path=worker_model_path,
        ppe_model_path=ppe_model_path,
        device='cuda'
    )
    safety_engine = SafetyRulesEngine(compliance_config_path)
    profiler = PerformanceProfiler(device='cuda')
    metrics_collector = MetricsCollector()

    profiler.start()

    # Prepare output video writer
    video_writer = VideoWriter(
        str(output_dir / 'annotated_video.mp4'),
        fps=video_processor.effective_fps,
        frame_size=(video_processor.width, video_processor.height)
    )

    # Frame-by-frame processing
    all_results = {}
    frame_count = 0

    logger.info("Starting inference...")

    for frame, frame_idx, metadata in video_processor:
        if max_frames and frame_count >= max_frames:
            break

        # Inference
        frame_detections = pipeline.process_frame(frame)

        # Compliance checking
        frame_compliance = {}
        for worker_id, ppe_items in frame_detections.items():
            compliance_result = safety_engine.evaluate_worker(worker_id, ppe_items, frame_idx)
            frame_compliance[worker_id] = compliance_result

        # Record metrics
        metrics_collector.record_frame(frame_idx, frame_detections, frame_compliance)

        # Performance profiling
        if pipeline.timings['total']:
            latency = pipeline.timings['total'][-1]
            profiler.record_frame(frame_idx, latency)

        # Draw annotations
        annotated_frame = draw_detections(frame, frame_detections, frame_compliance)
        video_writer.write_frame(annotated_frame)

        # Store frame results
        all_results[frame_idx] = {
            'detections': {
                str(wid): [
                    {
                        'class': item.class_name,
                        'confidence': item.confidence,
                        'bbox': item.bbox
                    }
                    for item in items
                ]
                for wid, items in frame_detections.items()
            },
            'compliance': {
                str(wid): {
                    'compliant': result.is_compliant,
                    'score': result.confidence_score,
                    'missing': result.missing_equipment
                }
                for wid, result in frame_compliance.items()
            }
        }

        frame_count += 1

        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count} frames...")

    video_writer.close()
    video_processor.close()

    logger.info(f"Processed {frame_count} total frames")

    # Generate reports
    logger.info("Generating reports...")

    # Performance report
    perf_stats = profiler.print_summary()
    profiler.export_json(str(output_dir / 'performance.json'))
    profiler.export_csv(str(output_dir / 'performance.csv'))

    # Compliance report
    video_metrics = safety_engine.evaluate_video(
        {frame_idx: frame_detections for frame_idx, frame_detections in enumerate(all_results)}
    )
    safety_engine.format_video_report(video_metrics)

    # Detection metrics
    metrics_summary = metrics_collector.print_summary()
    metrics_collector.export_json(str(output_dir / 'metrics.json'))
    metrics_collector.export_csv(str(output_dir / 'metrics.csv'))

    # Inference metrics
    inference_metrics = pipeline.get_metrics()
    with open(output_dir / 'inference_metrics.json', 'w') as f:
        json.dump(inference_metrics, f, indent=2)

    # Save frame-by-frame results
    with open(output_dir / 'frame_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate plots
    logger.info("Generating plots...")
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    try:
        profiler.plot_latency(str(plots_dir / 'latency.png'))
        profiler.plot_fps_distribution(str(plots_dir / 'fps_distribution.png'))
        profiler.plot_memory(str(plots_dir / 'memory.png'))
    except Exception as e:
        logger.warning(f"Could not generate profiler plots: {e}")

    try:
        metrics_collector.plot_class_detection_rates(str(plots_dir / 'detection_rates.png'))
        metrics_collector.plot_compliance_rate_over_time(str(plots_dir / 'compliance_timeline.png'))
    except Exception as e:
        logger.warning(f"Could not generate metrics plots: {e}")

    # Summary report
    report = {
        'video_info': video_processor.info(),
        'frames_processed': frame_count,
        'performance': {
            'mean_fps': perf_stats.mean_fps,
            'mean_latency_ms': perf_stats.mean_latency_ms,
            'p95_latency_ms': perf_stats.p95_latency_ms,
            'mean_memory_mb': perf_stats.mean_memory_mb,
            'peak_memory_mb': perf_stats.peak_memory_mb,
        },
        'compliance': {
            'avg_compliance_rate': video_metrics['avg_compliance_rate'],
            'total_alerts': video_metrics['total_alerts'],
            'critical_alerts': video_metrics['critical_alerts'],
        },
        'detection': {
            'total_workers': metrics_summary.total_workers,
            'total_ppe_detections': metrics_summary.total_ppe_detections,
            'avg_workers_per_frame': metrics_summary.avg_workers_per_frame,
        }
    }

    with open(output_dir / 'summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Inference complete. Results saved to {output_dir}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Run local video inference with compliance checking'
    )
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--worker-model', required=True, help='Path to worker detection model')
    parser.add_argument('--ppe-model', required=True, help='Path to PPE detection model')
    parser.add_argument('--compliance-config', default='config/compliance_rules.yaml',
                       help='Path to compliance rules YAML')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for processing')
    parser.add_argument('--max-frames', type=int, help='Limit number of frames (for testing)')

    args = parser.parse_args()

    try:
        report = process_video(
            video_path=args.video,
            worker_model_path=args.worker_model,
            ppe_model_path=args.ppe_model,
            compliance_config_path=args.compliance_config,
            output_dir=args.output,
            target_fps=args.fps,
            max_frames=args.max_frames
        )

        # Print summary
        print("\n" + "="*70)
        print("INFERENCE SUMMARY")
        print("="*70)
        print(f"FPS:                  {report['performance']['mean_fps']:.1f}")
        print(f"Latency (mean):       {report['performance']['mean_latency_ms']:.1f}ms")
        print(f"Compliance Rate:      {report['compliance']['avg_compliance_rate']:.1%}")
        print(f"Workers Detected:     {report['detection']['total_workers']}")
        print(f"Output Directory:     {args.output}")
        print("="*70 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
