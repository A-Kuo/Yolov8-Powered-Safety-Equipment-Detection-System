# Phase 3: Local Testing & Optimization
## YOLOv8 Safety Detection System

**Status:** Infrastructure Complete - Ready for Testing  
**Date:** April 2026  
**Previous Phase:** ✅ Phase 1-2 (Models trained & real data infrastructure)  

---

## Overview

Phase 3 transforms trained models into a production-ready system through:
1. **Local Video Testing** — Validate on real warehouse footage
2. **Performance Measurement** — FPS, latency, memory profiling
3. **Compliance Verification** — Safety rule enforcement
4. **Optimization Planning** — Identify bottlenecks for Phase 3.2

### Phase 3 Sub-Phases

| Phase | Week | Deliverable | Success Metric |
|-------|------|-------------|-----------------|
| **3.1** | Week 1 | Baseline profiling on 5-min video | FPS measured, bottleneck identified |
| **3.2** | Week 2 | Optimizations applied (FP16, ONNX) | FPS improvement ≥1.5x |
| **3.3** | Week 3 | Edge deployment prep | Validation on Snapdragon target |

---

## Infrastructure Overview

### Components Implemented ✅

```
Phase 3 Architecture:

Video Input
    ↓
[VideoProcessor]          (Extract frames, handle FPS resampling)
    ↓
[InferencePipeline]       (Worker detection → PPE detection)
    ↓
[PerformanceProfiler]     (Measure latency, FPS, memory)
    ↓
[SafetyRulesEngine]       (Evaluate compliance, generate alerts)
    ↓
[MetricsCollector]        (Track detections, confidence, compliance)
    ↓
[VideoWriter]             (Annotate with boxes & status)
    ↓
Output:
  - Annotated video
  - JSON results (detections, compliance)
  - Performance metrics (FPS, latency, memory)
  - Statistical plots (distributions, timelines)
  - Summary reports
```

### Files Created

**Core Modules:**
- `src/edge_deployment/video_processor.py` — Video I/O & frame extraction
- `src/edge_deployment/inference_pipeline.py` — Multi-model orchestration
- `src/edge_deployment/safety_rules_engine.py` — PPE compliance validation
- `src/edge_deployment/performance_profiler.py` — Performance measurement
- `src/edge_deployment/metrics_collector.py` — Statistics collection

**Scripts:**
- `scripts/run_local_video_inference.py` — End-to-end test pipeline

**Configuration:**
- `config/compliance_rules.yaml` — Safety policy definitions
- `config/performance_thresholds.yaml` — Optimization targets

---

## Quick Start

### 1. Prepare a Test Video

Choose one of:
- **Real warehouse video** — Your own footage (recommended)
- **Sample video** — Download from public dataset
- **Synthetic video** — Generate test data

```bash
# Check video info
python -c "
from src.edge_deployment.video_processor import get_video_info
info = get_video_info('path/to/video.mp4')
print(info)
"
```

### 2. Run Full Pipeline

```bash
python scripts/run_local_video_inference.py \
    --video path/to/warehouse_video.mp4 \
    --worker-model models/yolo/ppe_detector.pt \
    --ppe-model models/yolo/ppe_detector.pt \
    --output results/ \
    --fps 30
```

**Outputs:**
- `results/annotated_video.mp4` — Visual detections
- `results/summary_report.json` — Key metrics
- `results/performance.json` — FPS, latency, memory
- `results/plots/` — Performance charts

### 3. Review Results

```bash
# Print summary report
cat results/summary_report.json

# View performance metrics
cat results/performance.json

# Open annotated video
mpv results/annotated_video.mp4
```

---

## Module Details

### VideoProcessor
**Purpose:** Load and stream video frames

```python
from src.edge_deployment.video_processor import VideoProcessor

# Load video
processor = VideoProcessor('warehouse.mp4', target_fps=30)

# Iterate through frames
for frame, frame_idx, metadata in processor:
    # Process frame
    pass

# Or get specific frame
frame, metadata = processor.get_frame(50)
```

**Features:**
- Handles MP4, AVI, MOV (any OpenCV format)
- FPS resampling (skip frames to target FPS)
- Optional resolution scaling
- Memory-efficient streaming (no full load)

### InferencePipeline
**Purpose:** Orchestrate worker detection + PPE detection

```python
from src.edge_deployment.inference_pipeline import InferencePipelineWithMetrics

# Initialize
pipeline = InferencePipelineWithMetrics(
    worker_model_path='models/yolo/ppe_detector.pt',
    ppe_model_path='models/yolo/ppe_detector.pt',
    device='cuda'
)

# Warmup (avoid startup latency)
pipeline.warmup()

# Process frame
detections = pipeline.process_frame(frame)
# Returns: {worker_id: [DetectionResult]}

# Get performance metrics
metrics = pipeline.get_metrics()
print(f"FPS: {metrics['fps']:.1f}")
print(f"Latency: {metrics['total_ms']['mean']:.1f}ms")
```

**Features:**
- Multi-model orchestration
- Per-worker PPE crops (efficiency)
- Built-in latency tracking
- Warmup functionality

### SafetyRulesEngine
**Purpose:** Evaluate PPE compliance policy

```python
from src.edge_deployment.safety_rules_engine import SafetyRulesEngine

# Load policy
engine = SafetyRulesEngine('config/compliance_rules.yaml')

# Check worker compliance
result = engine.evaluate_worker(
    worker_id=0,
    detections=[safety_glasses, hard_hat, hi_vis_vest, work_boots]
)

# Result includes:
# - is_compliant (bool)
# - missing_equipment (list)
# - alerts (list of Alert objects)
# - confidence_score (% of required items detected)
```

**Features:**
- Policy-driven (all settings in YAML)
- Multiple alert severity levels
- Per-worker compliance scoring
- Video-level aggregation

### PerformanceProfiler
**Purpose:** Measure FPS, latency, memory usage

```python
from src.edge_deployment.performance_profiler import PerformanceProfiler
import time

# Initialize
profiler = PerformanceProfiler(device='cuda')
profiler.start()

# For each frame:
for frame in frames:
    t0 = time.time()
    results = pipeline.process_frame(frame)
    latency = time.time() - t0
    
    profiler.record_frame(frame_idx, latency)

# Get statistics
stats = profiler.get_stats()
print(f"Mean FPS:       {stats.mean_fps:.1f}")
print(f"Mean Latency:   {stats.mean_latency_ms:.1f}ms")
print(f"P95 Latency:    {stats.p95_latency_ms:.1f}ms")
print(f"Memory (Peak):  {stats.peak_memory_mb:.0f}MB")

# Generate plots
profiler.plot_latency('latency.png')
profiler.plot_fps_distribution('fps_dist.png')
profiler.print_summary()
```

**Features:**
- Per-frame latency tracking
- Memory profiling (CPU & GPU)
- Statistical aggregation (mean, median, p95, p99)
- Plot generation (requires matplotlib)
- JSON/CSV export

### MetricsCollector
**Purpose:** Track detection and compliance statistics

```python
from src.edge_deployment.metrics_collector import MetricsCollector

# Initialize
collector = MetricsCollector()

# For each frame:
collector.record_frame(
    frame_idx=0,
    frame_detections={0: [safety_glasses, hard_hat, ...]},
    compliance_results={0: compliance_result}
)

# Get summary
summary = collector.get_summary()
print(f"Total workers:   {summary.total_workers}")
print(f"Avg compliance:  {summary.avg_compliance_rate:.1%}")
print(f"PPE detections:  {summary.total_ppe_detections}")

# Class-specific metrics
for class_name, metrics in summary.class_metrics.items():
    print(f"{class_name}: detected in {metrics.detection_rate:.1%} of frames")

# Generate plots
collector.plot_class_detection_rates('detection_rates.png')
collector.plot_compliance_rate_over_time('compliance_timeline.png')
```

**Features:**
- Per-class confidence distributions
- Detection rate tracking
- Compliance statistics
- Frame-level metrics
- Visualization (requires matplotlib)

---

## Phase 3.1: Baseline Profiling

### Objective
Measure current performance without optimizations to establish baseline.

### Steps

1. **Find Test Video**
   ```bash
   # Option A: Use your own warehouse video
   # Option B: Download sample construction video from Roboflow
   # Option C: Create synthetic video with test script
   ```

2. **Run Baseline**
   ```bash
   python scripts/run_local_video_inference.py \
       --video test_video.mp4 \
       --worker-model models/yolo/ppe_detector.pt \
       --ppe-model models/yolo/ppe_detector.pt \
       --output baseline_results/ \
       --fps 30 \
       --max-frames 300
   ```

3. **Analyze Results**
   ```bash
   # Check summary
   cat baseline_results/summary_report.json
   
   # Key metrics to record:
   # - mean_fps (current throughput)
   # - mean_latency_ms (per-frame time)
   # - p95_latency_ms (worst case)
   # - peak_memory_mb (resource usage)
   ```

4. **Identify Bottleneck**
   - Is it worker detection or PPE detection?
   - Check `results/inference_metrics.json`:
     ```json
     {
       "worker_detection_ms": {"mean": 30},
       "ppe_detection_ms": {"mean": 25},
       "total_ms": {"mean": 60}
     }
     ```
   - If worker_detection >> ppe_detection → focus on worker optimizer
   - If ppe_detection >> worker_detection → focus on PPE optimizer

### Expected Baseline Results

On Intel Arc 140V with ppe_detector.pt (6.3MB):
```
FPS:              12-15 FPS (current)
Latency (mean):   65-85ms
Latency (p95):    90-120ms
Memory (peak):    1800MB
Status:           Below 30 FPS target, optimization needed
```

### Success Criteria for 3.1
- ✓ Baseline metrics recorded
- ✓ Bottleneck identified (worker vs PPE)
- ✓ Output video generated correctly
- ✓ Compliance checks working

---

## Phase 3.2: Quick Optimizations

### Optimization Priority

**#1: FP16 Precision** (Recommended first)
- Expected speedup: 1.5x
- Accuracy loss: <1%
- Implementation: Automatic in PyTorch

```python
# Enable FP16
import torch
model = torch.load('model.pt')
model.half()  # Convert to FP16
```

**#2: ONNX Runtime** (Cross-platform validation)
- Expected speedup: 1.1x over PyTorch
- Same accuracy
- Implementation: Export + inference

```bash
# Export to ONNX (from Colab)
python -c "
from ultralytics import YOLO
model = YOLO('models/yolo/ppe_detector.pt')
model.export(format='onnx', opset=12)
"
```

**#3: Resolution Reduction** (Trade-off)
- 480x480 instead of 640x640: 1.5x speedup, ~2% accuracy loss
- 320x320 instead of 640x640: 4x speedup, ~5% accuracy loss

```python
pipeline = InferencePipelineWithMetrics(...)
# Add parameter: input_size=480 (in pipeline)
```

**#4: Batch Processing** (GPU-only)
- Process 2-4 frames in parallel
- Expected speedup: 1.3-1.8x (GPU-bound)
- No benefit if CPU-bound

---

## Phase 3.3: Edge Preparation

### Snapdragon/Rubik Pi Validation

1. **ONNX Model Testing**
   ```bash
   # Export from Phase 2C training
   python scripts/run_local_video_inference.py \
       --video test.mp4 \
       --worker-model models/onnx/ppe_detector.onnx \
       --ppe-model models/onnx/ppe_detector.onnx \
       --output onnx_results/
   ```

2. **QNN Conversion** (Optional, Phase 4)
   ```bash
   # Convert ONNX to QNN DLC format
   # (Requires Qualcomm QNN SDK on edge device)
   ```

3. **Cross-compile Testing** (Future)
   - Test on actual Snapdragon hardware
   - Measure FPS/memory on device

### Success Criteria for 3.3
- ✓ ONNX inference validated
- ✓ FPS improvement ≥1.5x from baseline
- ✓ Memory usage <2GB
- ✓ Accuracy impact <2% mAP@50

---

## Compliance Rules Configuration

The safety policy is defined in `config/compliance_rules.yaml`:

```yaml
required_ppe:
  eye_protection:        # Safety glasses OR goggles
  head_protection:       # Hard hat (NOT regular hat)
  torso_protection:      # Hi-vis vest
  foot_protection:       # Work boots (NOT regular shoes)

confidence_thresholds:
  global_min: 0.25
  equipment_specific:
    hard_hat: 0.30      # Head protection critical
    safety_glasses: 0.35 # Small object, high threshold
    # ... etc
```

**To customize:**
1. Edit `config/compliance_rules.yaml`
2. Adjust thresholds based on Phase 3.1 results
3. Add/remove required PPE items
4. Modify alert severity levels

---

## Troubleshooting

### Issue: Low FPS (<10)

**Possible Causes & Solutions:**
1. Model too large
   → Try 480x480 resolution: expect 1.5x speedup
2. GPU not being used
   → Check device: `device='cuda'` in pipeline
3. CPU bottleneck
   → Profile worker vs PPE time separately
   → Optimize preprocessing (video reading)

### Issue: High Memory Usage (>2GB)

**Solutions:**
1. Reduce batch size (if using batching)
2. Use smaller model variant (nano instead of medium)
3. Reduce video resolution
4. Switch to ONNX (more efficient)

### Issue: Incorrect Detections

**Validation Steps:**
1. Check confidence threshold in `config/compliance_rules.yaml`
2. Compare with Phase 2C validation metrics
3. Verify model was trained on similar data
4. Look at `results/frame_results.json` for specific failures

### Issue: Compliance Alerts Not Generated

**Check:**
1. Video has workers detected (check `results/summary_report.json`)
2. Config has required PPE items (check `config/compliance_rules.yaml`)
3. Review detection confidence vs thresholds
4. Check frame-by-frame results in JSON

---

## Output Files Reference

| File | Purpose | Format |
|------|---------|--------|
| `annotated_video.mp4` | Visual results with boxes | Video |
| `frame_results.json` | Per-frame detections | JSON |
| `metrics.json` | Detection statistics | JSON |
| `performance.json` | FPS, latency, memory | JSON |
| `inference_metrics.json` | Model timing breakdown | JSON |
| `summary_report.json` | Executive summary | JSON |
| `plots/latency.png` | Latency timeline | PNG |
| `plots/fps_distribution.png` | FPS histogram | PNG |
| `plots/memory.png` | Memory usage timeline | PNG |
| `plots/detection_rates.png` | Class detection rates | PNG |
| `plots/compliance_timeline.png` | Compliance rate over time | PNG |

---

## Next Steps

### Immediate (Week 1)
- [ ] Prepare test video
- [ ] Run Phase 3.1 baseline profiling
- [ ] Record metrics in spreadsheet
- [ ] Identify optimization target

### Phase 3.2 (Week 2)
- [ ] Apply FP16 optimization
- [ ] Validate ONNX model
- [ ] Test resolution reduction
- [ ] Measure speedup vs accuracy

### Phase 3.3 (Week 3)
- [ ] ONNX cross-platform testing
- [ ] QNN conversion setup
- [ ] Edge device integration (if available)
- [ ] Final validation

### Phase 4 (Future)
- [ ] Quantization (INT8)
- [ ] QNN deployment
- [ ] Real-time monitoring dashboard
- [ ] Multi-camera tracking

---

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Qualcomm QNN SDK](https://docs.qualcomm.com/bundle/qnn-sdk)
- [PyTorch Mixed Precision](https://pytorch.org/docs/stable/amp.html)

---

**Status:** ✅ Phase 3 Infrastructure Complete  
**Next Checkpoint:** Phase 3.1 baseline profiling on real video  
**Timeline:** 3 weeks (Phase 3.1/3.2/3.3)
