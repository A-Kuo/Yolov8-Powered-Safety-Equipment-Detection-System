#!/usr/bin/env python3
"""
Dual-Backend Comparison: Local YOLOv8 vs Roboflow Cloud PPE Detection

Runs BOTH backends on the same test frames and compares:
- Detection accuracy (what PPE each model detects)
- Latency (inference time)
- Compliance verdicts (do they agree?)
- Per-class metrics (hard_hat, vest, glasses, boots)

Usage:
    python scripts/compare_inference_backends.py \
        --video test.mp4 \
        --local-model models/yolo/ppe_detector.pt \
        --output results/comparison/
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.edge_deployment.video_processor import VideoProcessor
from src.edge_deployment.inference_pipeline import InferencePipelineWithMetrics
from src.edge_deployment.safety_rules_engine import SafetyRulesEngine


def run_comparison(video_path: str,
                   local_model_path: str,
                   output_dir: str,
                   max_frames: int = 100,
                   use_roboflow: bool = True) -> dict:
    """
    Run BOTH inference backends on same video and compare results.

    Returns comparison report with accuracy metrics per class.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("DUAL-BACKEND COMPARISON: Local YOLOv8 vs Roboflow Cloud")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Frames to process: {max_frames}")
    print(f"Local model: {local_model_path}")
    print(f"Cloud backend: {'ENABLED' if use_roboflow else 'DISABLED'}\n")

    # ── Initialize both backends ────────────────────────────────────────────
    video = VideoProcessor(video_path, target_fps=30)

    local_pipeline = InferencePipelineWithMetrics(
        worker_model_path=local_model_path,
        ppe_model_path=local_model_path,
        device='cuda',
        use_roboflow=False,  # Force local
        fp16=True,
        input_size=480,
    )

    cloud_pipeline = None
    if use_roboflow:
        try:
            cloud_pipeline = InferencePipelineWithMetrics(
                worker_model_path=local_model_path,
                ppe_model_path=local_model_path,
                device='cuda',
                use_roboflow=True,  # Force cloud
                fp16=False,
            )
            print("✅ Roboflow backend initialized")
        except Exception as e:
            print(f"⚠️  Roboflow initialization failed: {e}")
            print("   Continuing with local-only comparison\n")
            cloud_pipeline = None

    compliance_engine = SafetyRulesEngine(
        str(project_root / "config" / "compliance_rules.yaml"),
        temporal_window=0,  # No smoothing for fair comparison
    )

    # ── Process frames with both backends ────────────────────────────────────
    comparison_results = {
        'frames_processed': 0,
        'local': defaultdict(list),
        'cloud': defaultdict(list),
        'frame_results': {},
    }

    frame_count = 0
    print(f"{'Frame':>5} | {'Local Latency':>12} | {'Cloud Latency':>12} | Agreement")
    print("-" * 70)

    for frame, frame_idx, _ in video:
        if frame_count >= max_frames:
            break

        # ── Local inference ────────────────────────────────────────────────
        t0_local = time.perf_counter()
        local_detections = local_pipeline.process_frame(frame)
        latency_local_ms = (time.perf_counter() - t0_local) * 1000

        # ── Cloud inference ────────────────────────────────────────────────
        latency_cloud_ms = None
        cloud_detections = None
        if cloud_pipeline:
            t0_cloud = time.perf_counter()
            try:
                cloud_detections = cloud_pipeline.process_frame(frame)
                latency_cloud_ms = (time.perf_counter() - t0_cloud) * 1000
            except Exception as e:
                latency_cloud_ms = None
                cloud_detections = {}

        # ── Compliance checks ──────────────────────────────────────────────
        local_compliance = {}
        cloud_compliance = {}

        for wid, ppe_items in local_detections.items():
            result = compliance_engine.evaluate_worker(wid, ppe_items, frame_idx)
            local_compliance[wid] = result.is_compliant

        if cloud_detections:
            for wid, ppe_items in cloud_detections.items():
                result = compliance_engine.evaluate_worker(wid, ppe_items, frame_idx)
                cloud_compliance[wid] = result.is_compliant

        # ── Compare results ────────────────────────────────────────────────
        agreement = "✅" if local_compliance == cloud_compliance else "⚠️"
        latency_str = f"{latency_local_ms:.0f}ms" if latency_local_ms else "N/A"
        cloud_str = f"{latency_cloud_ms:.0f}ms" if latency_cloud_ms else "N/A"

        print(f"{frame_count:>5} | {latency_str:>12} | {cloud_str:>12} | {agreement}")

        # ── Store metrics ──────────────────────────────────────────────────
        comparison_results['frames_processed'] += 1
        comparison_results['local']['latencies'].append(latency_local_ms)
        if latency_cloud_ms:
            comparison_results['cloud']['latencies'].append(latency_cloud_ms)

        # Count detections per class
        for wid, ppe_items in local_detections.items():
            for item in ppe_items:
                comparison_results['local']['detections'].append(item.class_name)

        if cloud_detections:
            for wid, ppe_items in cloud_detections.items():
                for item in ppe_items:
                    comparison_results['cloud']['detections'].append(item.class_name)

        # Store frame-level results
        comparison_results['frame_results'][frame_count] = {
            'local_compliant': local_compliance,
            'cloud_compliant': cloud_compliance,
            'agreement': agreement == "✅",
            'local_latency_ms': round(latency_local_ms, 1) if latency_local_ms else None,
            'cloud_latency_ms': round(latency_cloud_ms, 1) if latency_cloud_ms else None,
        }

        frame_count += 1

    video.close()

    # ── Generate comparison report ─────────────────────────────────────────
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    print(f"\n[LATENCY ANALYSIS]")
    local_lats = comparison_results['local']['latencies']
    cloud_lats = comparison_results['cloud']['latencies']

    if local_lats:
        local_avg = np.mean(local_lats)
        local_p95 = np.percentile(local_lats, 95)
        print(f"  Local (FP16+480px):")
        print(f"    Avg latency:  {local_avg:.1f} ms/frame")
        print(f"    P95 latency:  {local_p95:.1f} ms/frame")
        print(f"    FPS:          {1000 / local_avg:.1f}")

    if cloud_lats:
        cloud_avg = np.mean(cloud_lats)
        cloud_p95 = np.percentile(cloud_lats, 95)
        speedup = cloud_avg / local_avg if local_lats else 1.0
        print(f"\n  Cloud (Roboflow API):")
        print(f"    Avg latency:  {cloud_avg:.1f} ms/frame")
        print(f"    P95 latency:  {cloud_p95:.1f} ms/frame")
        print(f"    FPS:          {1000 / cloud_avg:.1f}")
        print(f"\n  Speedup (local is X× faster): {speedup:.1f}×")

    print(f"\n[DETECTION ACCURACY]")
    local_detections = comparison_results['local']['detections']
    cloud_detections = comparison_results['cloud']['detections']

    # Count per-class detections
    local_counts = defaultdict(int)
    cloud_counts = defaultdict(int)
    for det in local_detections:
        local_counts[det] += 1
    for det in cloud_detections:
        cloud_counts[det] += 1

    all_classes = set(local_counts.keys()) | set(cloud_counts.keys())
    print(f"\n  Class detections (side-by-side):")
    print(f"  {'Class':<20} {'Local':<10} {'Cloud':<10}")
    print(f"  {'-'*40}")
    for cls in sorted(all_classes):
        local_cnt = local_counts.get(cls, 0)
        cloud_cnt = cloud_counts.get(cls, 0)
        diff = "✅" if local_cnt == cloud_cnt else "⚠️"
        print(f"  {cls:<20} {local_cnt:<10} {cloud_cnt:<10} {diff}")

    print(f"\n[COMPLIANCE AGREEMENT]")
    frame_results = comparison_results['frame_results']
    agreements = sum(1 for r in frame_results.values() if r['agreement'])
    agreement_pct = 100 * agreements / len(frame_results) if frame_results else 0
    print(f"  Frames with matching verdicts: {agreements}/{len(frame_results)} ({agreement_pct:.1f}%)")

    if agreement_pct < 100:
        print(f"\n  Disagreements:")
        for frame_idx, result in frame_results.items():
            if not result['agreement']:
                print(f"    Frame {frame_idx}: Local={result['local_compliant']}, "
                      f"Cloud={result['cloud_compliant']}")

    # ── Save detailed report ───────────────────────────────────────────────
    report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        'timestamp': datetime.now().isoformat(),
        'video': str(video_path),
        'frames_processed': comparison_results['frames_processed'],
        'local_model': str(local_model_path),
        'cloud_enabled': cloud_pipeline is not None,
        'latency': {
            'local_avg_ms': float(np.mean(local_lats)) if local_lats else None,
            'local_p95_ms': float(np.percentile(local_lats, 95)) if local_lats else None,
            'cloud_avg_ms': float(np.mean(cloud_lats)) if cloud_lats else None,
            'cloud_p95_ms': float(np.percentile(cloud_lats, 95)) if cloud_lats else None,
        },
        'accuracy': {
            'local_detections': dict(local_counts),
            'cloud_detections': dict(cloud_counts),
            'compliance_agreement_pct': round(agreement_pct, 1),
        },
        'frame_results': frame_results,
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n[OUTPUT]")
    print(f"  Report: {report_path}")
    print("\n" + "="*80 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Compare local YOLOv8 vs Roboflow cloud PPE detection'
    )
    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--local-model', default='models/yolo/ppe_detector.pt',
                        help='Path to local PPE model')
    parser.add_argument('--output', default='output/comparison/',
                        help='Output directory for reports')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Max frames to process')
    parser.add_argument('--no-roboflow', action='store_true',
                        help='Skip Roboflow (local-only comparison)')

    args = parser.parse_args()

    report = run_comparison(
        video_path=args.video,
        local_model_path=args.local_model,
        output_dir=args.output,
        max_frames=args.max_frames,
        use_roboflow=not args.no_roboflow,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
