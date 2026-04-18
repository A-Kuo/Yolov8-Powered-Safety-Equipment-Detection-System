# Dual-Pipeline Roadmap: Local + Roboflow Validation

**Objective:** Build a production-grade safety compliance system that detects PPE in real-time, with fallback between local edge inference and cloud validation.

**Timeline:** 1-2 hours (split across sessions)

---

## Architecture Overview

```
Camera Feed
    ↓
┌─────────────────────────────┐
│  DUAL INFERENCE PIPELINE    │
├─────────────────────────────┤
│ ┌───────────────────────┐   │
│ │  LOCAL YOLOv8 (Edge)  │   │  0ms cloud latency
│ │  Phase 2A Trained     │   │  ~40ms inference (GPU)
│ │  Hard-hat detection   │   │  Offline fallback
│ └───────────────────────┘   │
│           ↓                 │
│  ┌──────────────────────┐   │
│  │  Compliance Engine   │   │  Policy-driven rules
│  │  YAML configurable   │   │  Per-worker scoring
│  └──────────────────────┘   │
│           ↓                 │
│  ┌──────────────────────┐   │
│  │  Roboflow Cloud      │   │  Production accuracy
│  │  (Optional)          │   │  95%+ detection
│  │  8,700+ real images  │   │  Backup validation
│  └──────────────────────┘   │
└─────────────────────────────┘
         ↓
    ALERTS / COMPLIANCE
    REPORT
```

---

## Phase A: Local PPE Model Training (30-60 mins, Colab)

### Checkpoint A1: Prepare for Colab Training
- [ ] Verify Colab notebook exists: [link](https://colab.research.google.com/github/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System/blob/claude/setup-yolo-safety-detection-UoZTY/notebooks/Colab_Training_PPE_Detection.ipynb)
- [ ] Review Phase 2A training guide: `PHASE2A_EXECUTION_GUIDE.md`
- [ ] Confirm Google Drive is connected to Colab account

### Checkpoint A2: Run Phase 2A Training in Colab
**DO THIS IN COLAB:**
```
Open notebook → Runtime=T4 GPU → Run cells [1-13]
Expected: 30 mins, mAP@50 → 0.65-0.70
Output: best.pt file saved to Google Drive
```

### Checkpoint A3: Download & Deploy Model
**DO THIS LOCALLY:**
```bash
# Download best.pt from Google Drive
# Place it in models/yolo/ppe_detector.pt (overwrite COCO base model)

# Verify the trained model
python -c "
from ultralytics import YOLO
m = YOLO('models/yolo/ppe_detector.pt')
print('Model classes:', list(m.names.values())[:15])
print('File size:', m.model_path.stat().st_size / 1e6, 'MB')
"
```

---

## Phase B: Local Model Validation (10 mins)

### Checkpoint B1: Synthetic Test (No Camera Needed)
```bash
# Test edge inference with synthetic frames
python scripts/test_edge_inference.py --frames 60

# Expected output:
#   ✅ Compliance Pipeline Test — 4/4 scenarios pass
#   ✅ Edge Inference — 0 network connections
#   [Detection should now show PPE classes, not just generic objects]
```

### Checkpoint B2: Live Camera Test (If Hardware Available)
```bash
# Real-time inference on webcam
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/yolo/ppe_detector.pt \
    --ppe-model models/yolo/ppe_detector.pt \
    --fp16 \
    --input-size 480 \
    --temporal-smoothing 5 \
    --output results/local_validation/

# Press Q to quit, S to save snapshot
# Check results/local_validation/session_report_*.json
```

---

## Phase C: Roboflow Cloud Setup (5 mins)

### Checkpoint C1: Get Roboflow API Key
1. Visit: https://app.roboflow.com/settings/profile
2. Copy your API key
3. Set environment variable:
   ```bash
   export ROBOFLOW_API_KEY="your_key_here"
   ```

### Checkpoint C2: Verify Cloud Connection
```bash
python -c "
from inference_sdk import InferenceHTTPClient
client = InferenceHTTPClient(
    api_url='https://detect.roboflow.com',
    api_key='$ROBOFLOW_API_KEY'
)
print('✅ Roboflow connection OK')
"
```

---

## Phase D: Dual-Backend Comparison (10-15 mins)

### Checkpoint D1: Run Comparison Test
```bash
# Compare local vs cloud on same video
python scripts/compare_inference_backends.py \
    --video results/local_validation/annotated_*.mp4 \
    --local-model models/yolo/ppe_detector.pt \
    --output results/comparison/ \
    --max-frames 50

# Generates:
#   - Side-by-side accuracy table
#   - Latency comparison (40ms vs 150-200ms)
#   - Compliance agreement %
#   - Per-class detection counts
```

### Checkpoint D2: Analyze Results
```bash
# View comparison report
cat results/comparison/comparison_report_*.json | jq '.accuracy'

# Expected output:
# {
#   "local_detections": {
#     "hard_hat": 32,
#     "hi_vis_vest": 28,
#     "safety_glasses": 25,
#     "work_boots": 30
#   },
#   "cloud_detections": {
#     "hard_hat": 35,
#     "hi_vis_vest": 30,
#     "safety_glasses": 28,
#     "work_boots": 32
#   },
#   "compliance_agreement_pct": 96.0
# }
```

### Checkpoint D3: Live Comparison (Optional, Advanced)
```bash
# Run both backends simultaneously on webcam
# Terminal 1: Local
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/yolo/ppe_detector.pt \
    --ppe-model models/yolo/ppe_detector.pt \
    --fp16 --input-size 480 \
    --output results/local_live/

# Terminal 2: Cloud (in parallel)
python scripts/run_live_inference.py \
    --camera 0 \
    --use-roboflow \
    --output results/cloud_live/

# Compare reports side-by-side
diff -u results/local_live/session_report_*.json \
         results/cloud_live/session_report_*.json
```

---

## Phase E: Production Deployment Setup (Optional)

### Checkpoint E1: Select Strategy

**Option 1: Edge-First (Recommended)**
```bash
# Use local model for speed, Roboflow for validation
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/yolo/ppe_detector.pt \
    --ppe-model models/yolo/ppe_detector.pt \
    --fp16 \
    --input-size 480 \
    --temporal-smoothing 5 \
    --no-display \
    --output /var/log/ppe_monitoring/
```

**Option 2: Cloud-First (Accuracy Priority)**
```bash
python scripts/run_live_inference.py \
    --camera 0 \
    --use-roboflow \
    --output /var/log/ppe_monitoring/ \
    --no-display
```

**Option 3: Ensemble (Best of Both)**
```bash
# Run both, majority voting (TODO: implement)
# High confidence → use local (fast)
# Disagreement → alert supervisor (manual review)
```

### Checkpoint E2: Deploy to Edge Device
```bash
# For Snapdragon/Rubik Pi:
1. Export model to ONNX: scripts/convert_to_onnx.py
2. Compile with QNN: scripts/optimize_for_qnn.py
3. Deploy .dlc file to target device
```

---

## Validation Checklist

### Local Model (Phase 2A)
- [ ] Training completed in Colab (mAP@50 > 0.65)
- [ ] best.pt downloaded from Google Drive
- [ ] Copied to models/yolo/ppe_detector.pt
- [ ] test_edge_inference.py shows PPE classes detected
- [ ] Live camera test runs at 25+ FPS (GPU) or 10+ FPS (CPU)
- [ ] session_report.json shows compliance verdicts correct
- [ ] Zero network connections in offline mode

### Roboflow Cloud
- [ ] API key obtained from app.roboflow.com
- [ ] ROBOFLOW_API_KEY environment variable set
- [ ] Test connection succeeds (no auth errors)
- [ ] run_live_inference.py --use-roboflow works
- [ ] Cloud detections show higher accuracy than Phase 2A
- [ ] Latency 150-200ms acceptable for non-critical path

### Dual-Pipeline Comparison
- [ ] compare_inference_backends.py runs without error
- [ ] Side-by-side accuracy table generated
- [ ] Compliance agreement >95% (most frames agree)
- [ ] Latency speedup: local is 3-4× faster than cloud
- [ ] Per-class detection counts reasonable
- [ ] JSON report saved with full metrics

### End-to-End System
- [ ] Compliance engine fires correct alerts
- [ ] Temporal smoothing prevents false alerts
- [ ] Hard hat detection works (CRITICAL)
- [ ] Vest detection works (WARNING)
- [ ] Proper alert severity levels assigned
- [ ] System handles missing workers gracefully
- [ ] No crashes on edge cases (no detections, empty frames)

---

## Expected Performance Metrics

### Phase 2A Local Model (GPU)
| Metric | Target | Typical |
|--------|--------|---------|
| Hard hat accuracy | 80%+ | 90% |
| Vest accuracy | 75%+ | 88% |
| Glasses accuracy | 70%+ | 82% |
| Boots accuracy | 65%+ | 76% |
| Latency | <50ms | 40ms |
| FPS | 20+ | 25 |
| Network calls | 0 | 0 ✅ |

### Roboflow Cloud
| Metric | Target | Typical |
|--------|--------|---------|
| Hard hat accuracy | 90%+ | 95% |
| Vest accuracy | 88%+ | 93% |
| Glasses accuracy | 85%+ | 90% |
| Boots accuracy | 80%+ | 87% |
| Latency | 150-200ms | 180ms |
| FPS | 5-10 | 7 |
| Network calls | Required | Required |

---

## Troubleshooting Matrix

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| No PPE detections in test | Still using COCO base model | Verify file size (trained ~50MB, base ~25MB) |
| Colab training OOM | Batch size too large | Reduce batch=16 → batch=4 |
| Roboflow API key error | Invalid key or expired | Regenerate from app.roboflow.com |
| High compliance disagreement | Models detecting different objects | Retrain Phase 2A, or use Roboflow cloud |
| Latency >200ms local | No GPU available | Check CUDA installed, torch.cuda.is_available() |
| No network connection | ROBOFLOW_API_KEY not set | `export ROBOFLOW_API_KEY="your_key"` |

---

## Success Criteria

✅ **System is production-ready when:**
1. Phase 2A model detects hard_hat, vest, glasses, boots >80% accuracy
2. Roboflow validates with >90% accuracy on same test set
3. Compliance engine correctly identifies compliant/non-compliant workers
4. Local model runs at 25+ FPS on GPU (or 10+ on CPU)
5. Roboflow provides fallback when offline not acceptable
6. Agreement between backends >95%
7. Zero crashes on edge cases
8. Full audit trail in JSON reports

---

## File Structure After Completion

```
Yolov8-Powered-Safety-Equipment-Detection-System/
├── models/
│   └── yolo/
│       └── ppe_detector.pt          ← Phase 2A trained model
├── scripts/
│   ├── test_edge_inference.py       ← Compliance + inference test
│   ├── compare_inference_backends.py ← Dual-pipeline comparison
│   ├── run_live_inference.py        ← Real-time monitoring
│   └── run_local_video_inference.py ← Batch processing
├── results/
│   ├── local_validation/            ← Phase B results
│   ├── comparison/                  ← Phase D results
│   └── edge_test/                   ← Phase B test reports
├── PHASE2A_EXECUTION_GUIDE.md       ← Training instructions
└── DUAL_PIPELINE_ROADMAP.md         ← This file
```

---

## Next Steps

### Immediate (Do Now)
- [ ] Run Phase 2A training in Colab (30 mins)
- [ ] Download trained model
- [ ] Deploy to models/yolo/ppe_detector.pt
- [ ] Run test_edge_inference.py to verify

### Short Term (This Week)
- [ ] Get Roboflow API key
- [ ] Run comparison script
- [ ] Analyze side-by-side results
- [ ] Document accuracy gap

### Medium Term (Next Week)
- [ ] Phase 2C: Fine-tune on real+synthetic data (60 epochs)
- [ ] Expected mAP@50 → 0.80+
- [ ] Re-run comparison

### Long Term (Production)
- [ ] Deploy to edge devices (Snapdragon/Rubik Pi)
- [ ] Set up monitoring dashboard
- [ ] Configure Roboflow as cloud backup
- [ ] Multi-camera tracking with persistent IDs

---

**Last Updated:** April 2026  
**Maintainer:** A-Kuo  
**Status:** Ready for Phase 2A Execution
