#!/usr/bin/env python3
"""
Real-time live camera / RTSP stream inference with PPE compliance checking.

Reads frames from a webcam or network stream, runs worker + PPE detection,
overlays compliance status, and displays the result live (or saves to file).

Usage — webcam:
    python scripts/run_live_inference.py --camera 0

Usage — RTSP stream:
    python scripts/run_live_inference.py --camera rtsp://192.168.1.100:554/live

Usage — Roboflow cloud backend:
    ROBOFLOW_API_KEY=rf-xxx python scripts/run_live_inference.py --camera 0 --use-roboflow

Usage — optimised local models:
    python scripts/run_live_inference.py \
        --camera 0 \
        --worker-model models/yolo/worker.pt \
        --ppe-model    models/yolo/ppe.pt \
        --fp16 \
        --input-size 480

Keys while running:
    Q / ESC  — quit
    S        — save current frame to output directory
    R        — reset compliance statistics

Output (optional, --output dir):
    output/annotated_<timestamp>.mp4
    output/session_report.json
"""

import argparse
import logging
import json
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.edge_deployment.video_processor import CameraProcessor, VideoWriter
from src.edge_deployment.inference_pipeline import InferencePipelineWithMetrics
from src.edge_deployment.safety_rules_engine import SafetyRulesEngine, AlertSeverity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Colour palette (BGR for OpenCV)
_COLOURS = {
    'compliant':     (0, 200, 0),
    'non_compliant': (0, 0, 220),
    'uncertain':     (0, 140, 255),
    'fps_box':       (30, 30, 30),
    'fps_text':      (220, 220, 220),
}


def annotate_frame(
    frame: np.ndarray,
    detections: dict,
    compliance_results: dict,
    fps: float,
    frame_idx: int,
) -> np.ndarray:
    """Draw PPE boxes, compliance status, and FPS counter onto frame."""
    out = frame.copy()

    for worker_id, ppe_items in detections.items():
        compliance = compliance_results.get(worker_id)
        if compliance is None:
            colour = _COLOURS['uncertain']
            label  = "? UNCERTAIN"
        elif compliance.is_compliant:
            colour = _COLOURS['compliant']
            label  = f"W{worker_id} COMPLIANT  {compliance.confidence_score:.0%}"
        else:
            colour = _COLOURS['non_compliant']
            missing = ", ".join(compliance.missing_equipment[:2])
            label  = f"W{worker_id} MISSING: {missing}"

        for item in ppe_items:
            x1, y1, x2, y2 = (int(v) for v in item.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                out, f"{item.class_name} {item.confidence:.2f}",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )

        # Worker compliance banner at top of frame
        h = out.shape[0]
        y_banner = min(20 + worker_id * 22, h - 5)
        cv2.putText(
            out, label,
            (8, y_banner),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA,
        )

    # FPS counter bottom-right
    fps_text = f"FPS {fps:.1f}  frame {frame_idx}"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    h, w = out.shape[:2]
    cv2.rectangle(out, (w - tw - 14, h - th - 12), (w - 2, h - 2), _COLOURS['fps_box'], -1)
    cv2.putText(
        out, fps_text,
        (w - tw - 10, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOURS['fps_text'], 1, cv2.LINE_AA,
    )

    return out


def run_live(
    camera_source,
    worker_model_path: str,
    ppe_model_path: str,
    compliance_config: str,
    output_dir: str,
    target_fps: int,
    use_roboflow: bool,
    fp16: bool,
    input_size: int,
    temporal_smoothing: int,
    show_display: bool,
) -> dict:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Initialise ──────────────────────────────────────────────────────
    cam = CameraProcessor(camera_source, target_fps=target_fps)

    pipeline = InferencePipelineWithMetrics(
        worker_model_path=worker_model_path,
        ppe_model_path=ppe_model_path,
        device='cuda',
        use_roboflow=use_roboflow,
        fp16=fp16,
        input_size=input_size,
    )
    pipeline.warmup(num_iterations=2)

    safety = SafetyRulesEngine(
        compliance_config,
        temporal_window=temporal_smoothing,
        alert_cooldown_frames=target_fps * 2,  # Suppress repeat alerts for 2 s
    )

    writer: VideoWriter | None = None
    save_video = output_dir is not None
    if save_video:
        out_path = output_dir / f"annotated_{timestamp}.mp4"
        writer = VideoWriter(str(out_path), fps=target_fps,
                             frame_size=(cam.width, cam.height))

    # ── Statistics ──────────────────────────────────────────────────────
    stats = {
        'frames': 0,
        'workers': 0,
        'compliant_frames': 0,
        'total_alerts': 0,
        'start_time': time.time(),
    }

    fps_history: list = []
    t_last = time.time()
    running = True

    def _quit(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _quit)
    signal.signal(signal.SIGTERM, _quit)

    logger.info(f"Live inference started — press Q/ESC to quit, S to save frame")

    # ── Main loop ───────────────────────────────────────────────────────
    for frame, frame_idx, _ in cam:
        if not running:
            break

        # Inference
        detections = pipeline.process_frame(frame)
        compliance_results = {}
        for wid, ppe_items in detections.items():
            compliance_results[wid] = safety.evaluate_worker(wid, ppe_items, frame_idx)

        # Update stats
        stats['frames'] += 1
        stats['workers'] += len(detections)
        if detections and all(r.is_compliant for r in compliance_results.values()):
            stats['compliant_frames'] += 1
        stats['total_alerts'] += sum(len(r.alerts) for r in compliance_results.values())

        # FPS calculation
        now = time.time()
        fps_history.append(1.0 / max(now - t_last, 1e-6))
        if len(fps_history) > 30:
            fps_history.pop(0)
        t_last = now
        current_fps = float(np.mean(fps_history))

        # Annotate
        annotated = annotate_frame(frame, detections, compliance_results, current_fps, frame_idx)

        if writer:
            writer.write_frame(annotated)

        if show_display:
            # Convert RGB → BGR for cv2.imshow
            display = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imshow("PPE Safety Monitor — Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
            elif key in (ord('s'), ord('S')):
                snap = output_dir / f"snapshot_{timestamp}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(snap), display)
                logger.info(f"Saved snapshot: {snap}")
            elif key in (ord('r'), ord('R')):
                pipeline.reset_metrics()
                stats = {k: (0 if isinstance(v, int) else v)
                         for k, v in stats.items()}
                logger.info("Statistics reset")

    # ── Cleanup & report ────────────────────────────────────────────────
    cam.close()
    if writer:
        writer.close()
    if show_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - stats['start_time']
    compliance_rate = stats['compliant_frames'] / max(stats['frames'], 1)
    inference_metrics = pipeline.get_metrics()

    report = {
        'session': {
            'timestamp': timestamp,
            'camera_source': str(camera_source),
            'duration_s': round(elapsed, 1),
            'frames_processed': stats['frames'],
            'backend': 'roboflow' if use_roboflow else 'local_yolov8',
            'fp16': fp16,
            'input_size': input_size,
        },
        'compliance': {
            'avg_compliance_rate': round(compliance_rate, 4),
            'total_alerts': stats['total_alerts'],
        },
        'performance': inference_metrics,
    }

    report_path = output_dir / f"session_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("LIVE INFERENCE SESSION COMPLETE")
    print("=" * 65)
    print(f"Duration:         {elapsed:.1f}s")
    print(f"Frames processed: {stats['frames']}")
    print(f"Compliance rate:  {compliance_rate:.1%}")
    print(f"Total alerts:     {stats['total_alerts']}")
    if inference_metrics:
        mean_ms = inference_metrics.get('total_ms', {}).get('mean', 0)
        print(f"Mean latency:     {mean_ms:.1f} ms/frame")
        print(f"Mean FPS:         {inference_metrics.get('fps', 0):.1f}")
    print(f"Report saved:     {report_path}")
    if writer:
        print(f"Video saved:      {out_path}")
    print("=" * 65 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Real-time PPE safety monitoring from camera or RTSP stream',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--camera', default=0,
                        help='Webcam device index (0, 1, …) or RTSP URL')
    parser.add_argument('--worker-model', default='',
                        help='Worker detector model path (not needed with --use-roboflow)')
    parser.add_argument('--ppe-model', default='',
                        help='PPE detector model path (not needed with --use-roboflow)')
    parser.add_argument('--compliance-config', default='config/compliance_rules.yaml',
                        help='Compliance rules YAML')
    parser.add_argument('--output', default='output/live/',
                        help='Directory for saved video and report')
    parser.add_argument('--fps', type=int, default=15,
                        help='Target capture/processing FPS')
    parser.add_argument('--use-roboflow', action='store_true',
                        help='Use Roboflow cloud workflow (requires ROBOFLOW_API_KEY)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable FP16 half-precision (local backend, GPU only)')
    parser.add_argument('--input-size', type=int, default=640,
                        help='Model input resolution (try 480 for ~1.5× speedup)')
    parser.add_argument('--temporal-smoothing', type=int, default=5,
                        help='Smooth compliance over last N frames (0 = off)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable live window (headless / server mode)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not write output video')

    args = parser.parse_args()

    # Convert camera arg: integer index if numeric, else keep as URL string
    camera_source = args.camera
    if isinstance(camera_source, str) and camera_source.isdigit():
        camera_source = int(camera_source)

    try:
        run_live(
            camera_source=camera_source,
            worker_model_path=args.worker_model,
            ppe_model_path=args.ppe_model,
            compliance_config=args.compliance_config,
            output_dir=args.output if not args.no_save else None,
            target_fps=args.fps,
            use_roboflow=args.use_roboflow,
            fp16=args.fp16,
            input_size=args.input_size,
            temporal_smoothing=args.temporal_smoothing,
            show_display=not args.no_display,
        )
        return 0
    except Exception as exc:
        logger.error(f"Live inference failed: {exc}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
