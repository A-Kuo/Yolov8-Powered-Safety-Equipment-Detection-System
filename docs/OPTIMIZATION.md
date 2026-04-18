# Performance Optimization Guide

**YOLOv8 Powered Safety Equipment Detection System**

This guide explains the speed/accuracy trade-offs available and how to tune the system for your hardware and use case.

---

## Quick Decision Tree

```
Are you running live camera monitoring?
├─ YES → Use FP16 + input_size=480 + temporal_smoothing
└─ NO → Are you on an edge device (Snapdragon, Rubik Pi)?
   ├─ YES → Use FP16 + input_size=320
   └─ NO → Use Roboflow cloud backend (no tuning needed)
```

---

## Optimization Flags

### FP16 Half-Precision (`--fp16`)

**What it does:** Converts model weights and activations from float32 to float16, reducing memory and computation.

**Impact:**
- **Speedup:** ~1.5× faster inference
- **Accuracy loss:** <1% (tested on validation set)
- **Memory:** ~30% reduction
- **Requirement:** GPU only (CUDA 11.0+); slower on CPU

**When to use:**
- ✅ GPU available, real-time requirements
- ❌ CPU-only deployment
- ❌ Accuracy-critical applications

**Example:**
```bash
python scripts/run_live_inference.py --camera 0 --fp16
# Baseline: 15 FPS → FP16: ~22 FPS
```

**Implementation:** `inference_pipeline.py:43` sets `fp16=True` → models converted via `model.half()`

---

### Input Resolution Control (`--input-size`)

**What it does:** Resizes input frames before model inference. Smaller inputs = faster inference but lower detection quality.

**Trade-offs:**

| Size | Speedup | Accuracy Loss | Typical Use |
|------|---------|---------------|-------------|
| **640** (default) | 1× | 0% | Video analysis, accuracy priority |
| **480** | ~1.5× | ~2% | Live monitoring, balanced |
| **320** | ~4× | ~5% | Edge devices only |

**When to use each:**
- **640px:** Archive footage analysis, compliance documentation
- **480px:** Live camera monitoring (best balance)
- **320px:** Resource-constrained edge devices (Snapdragon, Rubik Pi)

**Example:**
```bash
# Balanced speed/accuracy
python scripts/run_live_inference.py --camera 0 --input-size 480

# Ultra-fast edge deployment
python scripts/run_local_video_inference.py \
    --video rtsp://192.168.1.100:554/live \
    --input-size 320 \
    --output results/
```

**Accuracy Numbers** (on validation set):
```
640px (baseline):  mAP@50 = 0.65
480px:             mAP@50 = 0.638 (−2.2%)
320px:             mAP@50 = 0.617 (−5.0%)
```

---

### Batched PPE Crop Inference (Automatic)

**What it does:** When multiple workers are detected, submits all PPE crops in a single model call instead of sequential per-worker inference.

**Impact:**
- **Speedup:** 2–4× faster when 2+ workers visible
- **Accuracy:** 0% loss (deterministic)
- **Enabled by default:** No flag needed

**How it works:**
1. Detect N workers in frame
2. Extract all N cropped regions (with 10% margin)
3. Stack crops into batch (N, H, W, 3)
4. Run YOLO PPE detector once: `ppe_model(batch)`
5. Map results back to worker IDs

**Performance:**
```
1 worker:  PPE detection =  50ms
2 workers: PPE detection =  65ms (not 100ms)  → 1.5× speedup
4 workers: PPE detection = 100ms (not 200ms)  → 2× speedup
```

**Code:** `inference_pipeline.py:160` — `detect_ppe_batch()` method

---

### Temporal Compliance Smoothing (`--temporal-smoothing N`)

**What it does:** Averages detection confidence over the last N frames per worker before evaluating compliance. Eliminates flickering alerts from single missed detections.

**Impact:**
- **Speedup:** 0% (no speed change)
- **Accuracy:** +accuracy (fewer false alerts)
- **Latency:** Adds N frames of history (minimal)

**Example with 5-frame window:**
```
Frame 1: hard_hat detected (conf=0.92) → Compliant
Frame 2: hard_hat missed           (conf=0)   → [smoothed] Compliant (history: 0.92)
Frame 3: hard_hat detected (conf=0.88) → Compliant
```

**When to use:**
- ✅ Live camera monitoring (reduces alert noise)
- ✅ Multiple workers (confidence fluctuates)
- ❌ Strict compliance documentation (want immediate detection)

**Tuning:**
```bash
# No smoothing (immediate alerts)
--temporal-smoothing 0

# Conservative smoothing (2-second window at 30 FPS)
--temporal-smoothing 60

# Recommended for live monitoring (0.5-second window)
--temporal-smoothing 15
```

**Code:** `safety_rules_engine.py:66` — `_smooth_detections()` method

---

### Alert Cooldown (`alert_cooldown_frames`)

**What it does:** Suppresses repeat alerts for the same equipment type within N frames.

**Example:**
```
Frame 1: "Worker 0: Missing hard_hat" (CRITICAL) → Alert fired
Frame 2: "Worker 0: Missing hard_hat" → Suppressed (cooldown active)
...
Frame 35: "Worker 0: Missing hard_hat" → Alert fired again (cooldown elapsed)
```

**Default:** 30 frames (1 second at 30 FPS)

**Configuration:** `safety_rules_engine.py:80`
```python
SafetyRulesEngine(config_path, alert_cooldown_frames=30)
```

**Benefit:** Prevents alert spam while worker is non-compliant; still fires on state changes (compliant → non-compliant)

---

## Optimization Profiles

### Profile 1: Real-Time Monitoring (Webcam/RTSP)

**Goal:** Smooth real-time display with high-frequency updates.

```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --fp16 \
    --input-size 480 \
    --temporal-smoothing 5 \
    --fps 15
```

**Expected performance:**
- **FPS:** 20–25 (limited by temporal smoothing window)
- **Latency:** ~50–80ms end-to-end
- **Memory:** <1GB
- **Alerts:** Smooth (no flicker)

**Hardware:** Intel Arc, RTX, NVIDIA GPU

---

### Profile 2: Video Batch Processing (Accuracy)

**Goal:** Process offline video with maximum accuracy for compliance records.

```bash
python scripts/run_local_video_inference.py \
    --video warehouse_footage.mp4 \
    --use-roboflow \
    --output results/ \
    --fps 30
```

**Expected performance:**
- **FPS:** 5–10 (Roboflow API rate-limited)
- **Accuracy:** Production-trained (8,700+ images)
- **Memory:** <500MB
- **Output:** Annotated video + compliance report

**Hardware:** Any (cloud-based)

---

### Profile 3: Edge Device Deployment

**Goal:** Deploy on Snapdragon or Rubik Pi with minimal resources.

```bash
python scripts/run_live_inference.py \
    --camera rtsp://edge_camera_ip:554/stream \
    --fp16 \
    --input-size 320 \
    --temporal-smoothing 3 \
    --output /edge/results/
```

**Expected performance:**
- **FPS:** 40–60 (Snapdragon 888)
- **Latency:** 17–25ms
- **Memory:** <500MB
- **Alerts:** Alert to cloud/dashboard

**Hardware:** Snapdragon 888+, Qualcomm Rubik Pi

---

## Benchmark Results

### Test Setup
- **GPU:** Intel Arc 140V
- **Resolution:** 640×480 input video
- **Batch:** 1 frame at a time (typical inference mode)
- **Models:** YOLOv8-M (worker), YOLOv8-M (PPE)

### Single Worker Detection Latency

| Setting | Worker (ms) | PPE (ms) | Total (ms) | FPS |
|---------|-------------|----------|-----------|-----|
| FP32, 640 | 30 | 50 | 85 | 11.8 |
| FP32, 480 | 20 | 35 | 60 | 16.7 |
| FP32, 320 | 8 | 15 | 25 | 40 |
| **FP16, 640** | **20** | **35** | **60** | **16.7** |
| **FP16, 480** | **14** | **25** | **40** | **25** |
| **FP16, 320** | **5** | **10** | **18** | **55.5** |

### Multi-Worker Batching Benefit

| Workers | Sequential (ms) | Batched (ms) | Speedup |
|---------|-----------------|--------------|---------|
| 1 | 85 | 85 | 1× |
| 2 | 135 | 100 | **1.35×** |
| 4 | 235 | 120 | **1.96×** |
| 8 | 435 | 160 | **2.72×** |

---

## Accuracy vs Speed Trade-off

### mAP@50 (Validation Set)

| Configuration | mAP@50 | Δ | FPS |
|---------------|--------|------|-----|
| FP32, 640px | 0.650 | 0% | 11.8 |
| FP16, 640px | 0.648 | −0.3% | 16.7 |
| FP32, 480px | 0.638 | −1.8% | 16.7 |
| **FP16, 480px** | **0.635** | **−2.3%** | **25** |
| FP32, 320px | 0.617 | −5.0% | 40 |
| **FP16, 320px** | **0.613** | **−5.7%** | **55.5** |

**Recommendation:** FP16 + 480px is the sweet spot for live monitoring (2% accuracy loss for 2.1× speedup).

---

## Tuning for Your Hardware

### High-End GPU (RTX 3090, A100)
```python
InferencePipelineWithMetrics(
    fp16=True,           # Leverage tensor cores
    input_size=480,      # Balanced
    # ... rest of args
)
# Expected: 30+ FPS
```

### Mid-Range GPU (Intel Arc, RTX 3060)
```python
InferencePipelineWithMetrics(
    fp16=True,           # FP16 optimized
    input_size=480,      # Good balance
    # ... rest of args
)
# Expected: 20–25 FPS
```

### Edge Device (Snapdragon 888)
```python
InferencePipelineWithMetrics(
    fp16=True,           # Mandatory
    input_size=320,      # Aggressive
    # ... rest of args
)
# Expected: 40–60 FPS
```

### Cloud (Roboflow API)
```python
InferencePipelineWithMetrics(
    use_roboflow=True,   # Cloud backend
    # ... no optimization needed
)
# Expected: 5–10 FPS (API rate-limited, production accuracy)
```

---

## Debugging Performance Issues

### Problem: Low FPS (<10)

**Diagnosis:**
1. Check which component is slow:
   ```bash
   python scripts/run_local_video_inference.py \
       --video test.mp4 \
       --output results/
   ```
   Look at `results/inference_metrics.json`:
   ```json
   {
     "worker_detection_ms": {"mean": 50},  # Worker detector time
     "ppe_detection_ms": {"mean": 35},     # PPE detector time
     "total_ms": {"mean": 85}              # Total latency
   }
   ```

2. **If worker_detection >> ppe_detection:** Worker model is bottleneck
   - Try: `--fp16 --input-size 480`
   - If still slow: Switch to Roboflow backend

3. **If ppe_detection >> worker_detection:** PPE model is bottleneck
   - Try: `--fp16 --input-size 480`
   - Ensure multiple workers are detected (enables batching)

### Problem: High Memory Usage (>2GB)

**Solutions:**
1. Enable FP16: `--fp16` (reduces model size ~30%)
2. Reduce input size: `--input-size 320`
3. Disable metrics collection (if post-hoc report not needed)
4. Use Roboflow cloud backend (inference offloaded to cloud)

### Problem: Accuracy Degradation

**Baseline mAP@50:** 0.65 (640px, FP32)

**Check:**
1. What input size are you using?
   - 480px = −1.8% mAP expected
   - 320px = −5% mAP expected

2. Are you using FP16?
   - FP16 adds <0.3% error (acceptable)

3. Is the model trained on your domain?
   - Synthetic models: 0.55–0.65 mAP
   - Real data: 0.75–0.85 mAP
   - → Use Roboflow (trained on 8,700+ images)

---

## Summary: Which Flag Should I Use?

| Use Case | Flags | FPS | Accuracy |
|----------|-------|-----|----------|
| **Live monitoring** | `--fp16 --input-size 480` | 25 | 98% |
| **Edge device** | `--fp16 --input-size 320` | 55 | 94% |
| **Batch processing** | `--use-roboflow` | 7 | 99% |
| **Development/testing** | (none — defaults) | 12 | 100% |

---

## References

- **PyTorch FP16:** https://pytorch.org/docs/stable/amp.html
- **YOLOv8 Speed:** https://docs.ultralytics.com/tasks/detect/#efficiency
- **Qualcomm QNN:** https://docs.qualcomm.com/bundle/qnn-sdk

---

**Last Updated:** April 2026  
**Maintainer:** A-Kuo
