"""
Performance profiling for inference pipeline.

Phase 3: Measure FPS, latency, memory usage, and identify bottlenecks.
"""

import time
import psutil
import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for single frame inference."""
    frame_idx: int
    latency_ms: float
    fps: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None


@dataclass
class ProfilerStats:
    """Aggregated profiler statistics."""
    total_frames: int
    total_time_sec: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_fps: float
    mean_memory_mb: float
    peak_memory_mb: float
    mean_gpu_memory_mb: Optional[float] = None
    peak_gpu_memory_mb: Optional[float] = None


class PerformanceProfiler:
    """Profile inference pipeline performance."""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize profiler.

        Args:
            device: Device being profiled ('cuda', 'cpu', etc.)
        """
        self.device = device
        self.frame_metrics = []
        self.start_time = None
        self.process = psutil.Process()

        # GPU memory tracking (CUDA)
        self.gpu_available = False
        if device == 'cuda':
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
            except ImportError:
                pass

    def start(self):
        """Start profiling session."""
        self.start_time = time.time()
        self.frame_metrics = []

        if self.gpu_available:
            import torch
            torch.cuda.reset_peak_memory_stats()

        logger.info("Started profiling")

    def record_frame(self, frame_idx: int, latency_sec: float):
        """
        Record metrics for single frame.

        Args:
            frame_idx: Frame index
            latency_sec: Frame processing latency in seconds
        """
        latency_ms = latency_sec * 1000
        fps = 1.0 / latency_sec if latency_sec > 0 else 0

        # Memory metrics
        memory_mb = self.process.memory_info().rss / (1024 ** 2)

        gpu_memory_mb = None
        if self.gpu_available:
            try:
                import torch
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            except Exception:
                pass

        metrics = FrameMetrics(
            frame_idx=frame_idx,
            latency_ms=latency_ms,
            fps=fps,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb
        )

        self.frame_metrics.append(metrics)

    def get_stats(self) -> ProfilerStats:
        """Get aggregated statistics."""
        if not self.frame_metrics:
            raise RuntimeError("No metrics recorded")

        latencies = [m.latency_ms for m in self.frame_metrics]
        fps_values = [m.fps for m in self.frame_metrics]
        memory_values = [m.memory_mb for m in self.frame_metrics]

        gpu_memory_values = [m.gpu_memory_mb for m in self.frame_metrics if m.gpu_memory_mb is not None]

        total_time = sum(latencies) / 1000  # Convert to seconds
        total_frames = len(self.frame_metrics)

        stats = ProfilerStats(
            total_frames=total_frames,
            total_time_sec=total_time,
            mean_latency_ms=np.mean(latencies),
            median_latency_ms=np.median(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            min_latency_ms=np.min(latencies),
            max_latency_ms=np.max(latencies),
            mean_fps=np.mean(fps_values),
            mean_memory_mb=np.mean(memory_values),
            peak_memory_mb=np.max(memory_values),
            mean_gpu_memory_mb=np.mean(gpu_memory_values) if gpu_memory_values else None,
            peak_gpu_memory_mb=np.max(gpu_memory_values) if gpu_memory_values else None,
        )

        return stats

    def print_summary(self):
        """Print profiling summary."""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*70)

        print(f"\nFrames Processed: {stats.total_frames}")
        print(f"Total Time:       {stats.total_time_sec:.1f}s")

        print(f"\nLatency (ms):")
        print(f"  Mean:           {stats.mean_latency_ms:.1f}")
        print(f"  Median:         {stats.median_latency_ms:.1f}")
        print(f"  P95:            {stats.p95_latency_ms:.1f}")
        print(f"  P99:            {stats.p99_latency_ms:.1f}")
        print(f"  Min/Max:        {stats.min_latency_ms:.1f} / {stats.max_latency_ms:.1f}")

        print(f"\nThroughput (FPS):")
        print(f"  Mean:           {stats.mean_fps:.1f}")
        print(f"  Target:         30.0 (real-time)")
        if stats.mean_fps >= 30:
            print(f"  Status:         ✓ MEETS TARGET")
        elif stats.mean_fps >= 20:
            print(f"  Status:         ⚠ NEAR TARGET (optimization recommended)")
        else:
            print(f"  Status:         ✗ BELOW TARGET (optimization needed)")

        print(f"\nMemory (MB):")
        print(f"  Mean CPU:       {stats.mean_memory_mb:.0f}")
        print(f"  Peak CPU:       {stats.peak_memory_mb:.0f}")
        if stats.mean_gpu_memory_mb is not None:
            print(f"  Mean GPU:       {stats.mean_gpu_memory_mb:.0f}")
            print(f"  Peak GPU:       {stats.peak_gpu_memory_mb:.0f}")

        print("\n" + "="*70 + "\n")

        return stats

    def get_dataframe(self):
        """Get metrics as pandas DataFrame (if available)."""
        try:
            import pandas as pd
            return pd.DataFrame([asdict(m) for m in self.frame_metrics])
        except ImportError:
            logger.warning("pandas not installed, cannot create DataFrame")
            return None

    def export_json(self, output_path: str):
        """Export metrics to JSON file."""
        import json

        data = {
            'stats': asdict(self.get_stats()),
            'frames': [asdict(m) for m in self.frame_metrics]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported profiling data to {output_path}")

    def export_csv(self, output_path: str):
        """Export frame metrics to CSV."""
        import csv

        if not self.frame_metrics:
            return

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.frame_metrics[0]).keys())
            writer.writeheader()
            for m in self.frame_metrics:
                writer.writerow(asdict(m))

        logger.info(f"Exported frame metrics to {output_path}")

    def plot_latency(self, output_path: Optional[str] = None):
        """Plot latency over time."""
        try:
            import matplotlib.pyplot as plt

            if not self.frame_metrics:
                return

            frames = [m.frame_idx for m in self.frame_metrics]
            latencies = [m.latency_ms for m in self.frame_metrics]

            plt.figure(figsize=(12, 6))
            plt.plot(frames, latencies, 'b-', alpha=0.6)
            plt.axhline(y=33, color='g', linestyle='--', label='30 FPS target (33ms)')
            plt.xlabel('Frame Index')
            plt.ylabel('Latency (ms)')
            plt.title('Frame Latency Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if output_path:
                plt.savefig(output_path, dpi=100)
                logger.info(f"Saved latency plot to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")

    def plot_fps_distribution(self, output_path: Optional[str] = None):
        """Plot FPS distribution histogram."""
        try:
            import matplotlib.pyplot as plt

            if not self.frame_metrics:
                return

            fps_values = [m.fps for m in self.frame_metrics]

            plt.figure(figsize=(10, 6))
            plt.hist(fps_values, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(x=30, color='g', linestyle='--', linewidth=2, label='30 FPS target')
            plt.axvline(x=np.mean(fps_values), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(fps_values):.1f} FPS')
            plt.xlabel('FPS')
            plt.ylabel('Count')
            plt.title('FPS Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')

            if output_path:
                plt.savefig(output_path, dpi=100)
                logger.info(f"Saved FPS distribution plot to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")

    def plot_memory(self, output_path: Optional[str] = None):
        """Plot memory usage over time."""
        try:
            import matplotlib.pyplot as plt

            if not self.frame_metrics:
                return

            frames = [m.frame_idx for m in self.frame_metrics]
            cpu_memory = [m.memory_mb for m in self.frame_metrics]
            gpu_memory = [m.gpu_memory_mb for m in self.frame_metrics if m.gpu_memory_mb is not None]

            plt.figure(figsize=(12, 6))
            plt.plot(frames, cpu_memory, 'b-', label='CPU Memory', alpha=0.7)

            if gpu_memory:
                gpu_frames = [m.frame_idx for m in self.frame_metrics if m.gpu_memory_mb is not None]
                plt.plot(gpu_frames, gpu_memory, 'r-', label='GPU Memory', alpha=0.7)

            plt.xlabel('Frame Index')
            plt.ylabel('Memory (MB)')
            plt.title('Memory Usage Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if output_path:
                plt.savefig(output_path, dpi=100)
                logger.info(f"Saved memory plot to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")


def profile_inference_pipeline(pipeline, video_frames: List[np.ndarray], num_warmup: int = 5):
    """
    Profile inference pipeline on video frames.

    Args:
        pipeline: InferencePipeline instance
        video_frames: List of frames to profile
        num_warmup: Number of warmup frames to skip

    Returns:
        ProfilerStats
    """
    profiler = PerformanceProfiler(device=str(pipeline.device))
    profiler.start()

    # Warmup
    for i in range(min(num_warmup, len(video_frames))):
        _ = pipeline.process_frame(video_frames[i])

    # Profile
    for i in range(num_warmup, len(video_frames)):
        frame = video_frames[i]

        t0 = time.time()
        _ = pipeline.process_frame(frame)
        latency = time.time() - t0

        profiler.record_frame(i, latency)

    return profiler
