# Phase 2A: PPE Model Training & Dual-Pipeline Validation

**Goal:** Train a local PPE-detection model, then validate against Roboflow cloud backend.

---

## Step 1: Run Phase 2A Training in Colab (0.5 hours)

### 1.1 Open the Colab Notebook

```
https://colab.research.google.com/github/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System/blob/claude/setup-yolo-safety-detection-UoZTY/notebooks/Colab_Training_PPE_Detection.ipynb
```

### 1.2 Enable GPU

```
Runtime → Change runtime type → T4 GPU → Save
```

### 1.3 Run Training Cells

Execute cells **[1] → [13]** in order:
- [1-4]: Setup & imports
- [5-8]: Data preparation (synthetic dataset)
- [9.2]: **TRAINING** — 80 epochs on synthetic data
  - Expected time: ~30 mins on T4
  - Expected mAP@50: 0.65-0.70
- [10]: Model validation
- [11]: Export (PyTorch + ONNX + TorchScript)
- [12-13]: Upload to Google Drive

### 1.4 Download Trained Model

After training completes:
1. Find `/content/drive/MyDrive/YOLOv8_PPE_Training/runs/detect/train/weights/best.pt`
2. Download to your machine
3. Copy to: `models/yolo/ppe_detector.pt` (overwrite the COCO base model)

---

## Step 2: Validate Local Model on Edge Device

### 2.1 Test with Synthetic Webcam (No Hardware Needed)

```bash
# Synthetic validation (40 frames)
python scripts/test_edge_inference.py --frames 40

# Expected output:
#   Frames processed: 40
#   Hard hats detected: X (should detect synthetic hard hats now)
#   Vests detected: Y
#   Zero network connections ✅
#   Offline compliance checks working ✅
```

### 2.2 Test with Real Webcam (If Available)

```bash
# Live camera validation with trained model
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/yolo/ppe_detector.pt \
    --ppe-model models/yolo/ppe_detector.pt \
    --fp16 \
    --input-size 480 \
    --output results/phase2a_local/ \
    --fps 15

# Press Q to quit, S to save snapshots
# Check results/phase2a_local/session_report_*.json for metrics
```

---

## Step 3: Dual-Pipeline Comparison Test

### 3.1 Set Up Roboflow API Key

```bash
export ROBOFLOW_API_KEY="YOUR_API_KEY_FROM_ROBOFLOW"
# Get key from: https://app.roboflow.com/settings/profile
```

### 3.2 Run Comparison Script

```bash
# This runs BOTH backends on the same video and compares
python scripts/compare_inference_backends.py \
    --video warehouse_test.mp4 \
    --local-model models/yolo/ppe_detector.pt \
    --output results/comparison/

# Output will show:
#   Side-by-side accuracy comparison
#   Latency comparison (local FP16 vs Roboflow API)
#   Compliance results from each backend
#   Accuracy metrics per PPE class
```

### 3.3 Live Comparison (Optional)

Compare backends running simultaneously on webcam:

```bash
# Terminal 1: Local model
python scripts/run_live_inference.py \
    --camera 0 \
    --worker-model models/yolo/ppe_detector.pt \
    --ppe-model models/yolo/ppe_detector.pt \
    --fp16 --input-size 480 \
    --output results/local_live/

# Terminal 2: Roboflow cloud (in parallel)
python scripts/run_live_inference.py \
    --camera 0 \
    --use-roboflow \
    --output results/roboflow_live/
```

Then compare the JSON reports:
```bash
diff results/local_live/session_report_*.json results/roboflow_live/session_report_*.json
```

---

## Expected Results

### Phase 2A Local Model (Trained on 50 synthetic images, 80 epochs)

| Metric | Synthetic Baseline | Phase 2A Trained |
|--------|-------------------|------------------|
| mAP@50 | 0.559 | **0.65-0.70** |
| Hard hat detection | ❌ No | ✅ Yes (90%+) |
| Vest detection | ❌ No | ✅ Yes (88%+) |
| Latency (GPU) | ~85ms | ~40ms (FP16) |
| Network calls | 0 | **0** ✅ |
| Edge deployable | ✅ | ✅ (faster) |

### Roboflow Cloud (Pre-trained on 8,700+ real images)

| Metric | Roboflow |
|--------|----------|
| mAP@50 | **0.75-0.80** (production) |
| Hard hat detection | ✅ Yes (95%+) |
| Vest detection | ✅ Yes (93%+) |
| Latency (API) | 150-200ms |
| Network calls | **Required** |
| Edge deployable | ❌ (requires internet) |

### Combined Advantage

1. **Development:** Use local Phase 2A model → instant feedback, zero latency
2. **Production:** Use Roboflow for maximum accuracy → 95%+ detection
3. **Fallback:** If Roboflow API fails, local model still works offline
4. **Validation:** Compare predictions on same frames → measure accuracy gap

---

## Execution Timeline

| Step | Time | Output |
|------|------|--------|
| Phase 2A Colab training | 30 mins | `best.pt` (trained weights) |
| Download & copy model | 5 mins | `models/yolo/ppe_detector.pt` |
| Local validation | 5 mins | Test report showing PPE detection |
| Roboflow setup | 5 mins | API key configured |
| Comparison test | 10 mins | Side-by-side accuracy report |
| **Total** | **~55 mins** | **Full dual-pipeline working** |

---

## Troubleshooting

### Colab Training Issues

**OOM (out of memory):**
```python
# In Colab cell [9.2], reduce batch size:
results = model.train(
    data=yaml_path,
    epochs=80,
    imgsz=640,
    batch=4,  # ← reduce from 16 to 4
    device=0,
    patience=10,
    save=True,
    workers=4,
)
```

**Training too slow:**
- Increase batch size (if no OOM)
- Skip validation epochs: `val=False` (check after training)
- Use smaller model: `yolov8n.pt` instead of `yolov8m.pt`

### Local Model Issues

**No PPE detections in test:**
- Verify `models/yolo/ppe_detector.pt` is the trained model, not COCO base
- Check file size: trained model ~50MB, COCO base ~25MB
- Run: `python -c "from ultralytics import YOLO; m = YOLO('models/yolo/ppe_detector.pt'); print(m.names)"`
  - Should show custom classes, not COCO-80

**Roboflow API Key Issues**

```bash
# Verify key is set:
echo $ROBOFLOW_API_KEY

# Test connection:
python -c "
from inference_sdk import InferenceHTTPClient
client = InferenceHTTPClient(
    api_url='https://detect.roboflow.com',
    api_key='$ROBOFLOW_API_KEY'
)
print('Connected!')
"
```

---

## Success Criteria

After completing all steps, verify:

- [ ] Phase 2A training completed (mAP@50 > 0.65)
- [ ] Local model detects hard_hats, vests in test frames
- [ ] `test_edge_inference.py` shows PPE-specific detections
- [ ] Compliance engine fires correct alerts
- [ ] Zero network calls during local inference
- [ ] Roboflow API key working
- [ ] Roboflow detections more accurate than local (phase 2A)
- [ ] Comparison report generated
- [ ] Both backends produce compliance alerts correctly

**Final Result:** Production-ready dual-pipeline system:
- Local edge model for offline operation (0ms latency to cloud)
- Roboflow cloud for maximum accuracy (±95% on real-world footage)
- Automatic fallback if network fails

---

## Next Steps After Phase 2A

### Option A: Phase 2C Fine-tuning (Advanced)
1. Download Roboflow + Kaggle datasets (~1,250 real images)
2. Merge with synthetic data
3. Fine-tune Phase 2A model with lower learning rate
4. Expected: mAP@50 → 0.80+

### Option B: Production Deployment
1. Export Phase 2A model to ONNX for cross-platform
2. Deploy to Snapdragon/Rubik Pi with QNN
3. Use Roboflow as cloud backup

### Option C: Model Ensemble
1. Keep both local + Roboflow
2. Use majority voting on disagreements
3. Log confidence scores for QA

---

**Last Updated:** April 2026  
**Maintainer:** A-Kuo
