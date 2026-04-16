# YOLOv8 PPE Safety Detection — Training Notebooks

This directory contains Google Colab notebooks and guides for training the YOLOv8 safety detection models on cloud GPU.

## Files

### 📔 `Colab_Training_PPE_Detection.ipynb`

**Production-ready Colab notebook** for fine-tuning YOLOv8 on PPE detection dataset.

**Features:**
- [X.X] step indicators for clear execution flow
- Explicit PASS/FAIL status reporting
- Automatic GPU detection and setup
- Dataset auto-download from Ultralytics
- Transfer learning (freeze 10 backbone layers)
- ONNX export for edge deployment
- Automatic Google Drive backup

**Timeline:** ~3-4 hours on T4 GPU

**How to use:**
1. Open in Colab: https://colab.research.google.com/github/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System/blob/claude/setup-yolo-safety-detection-UoZTY/notebooks/Colab_Training_PPE_Detection.ipynb
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells [1] through [13] in order
4. Download models from Google Drive when complete

### 📋 `COLAB_QUICKSTART.md`

Quick reference guide (15 minutes) for getting started with Colab training.

**Sections:**
- Fast track setup (3 steps)
- Training timeline and expected results
- Optional API keys (Roboflow, Kaggle)
- Where results are saved (Google Drive paths)
- Troubleshooting common errors
- How to iterate and improve

### 📚 `docs/COLAB_SETUP.md`

Comprehensive setup guide with detailed explanations of each step.

**Includes:**
- GPU setup and verification
- Dependency installation
- Dataset download options
- Custom notebook creation
- Performance benchmarks
- Full code examples
- Next steps for deployment

## Getting Started (2 minutes)

### Option A: Use the provided notebook

```
1. Click: https://colab.research.google.com/github/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System/blob/claude/setup-yolo-safety-detection-UoZTY/notebooks/Colab_Training_PPE_Detection.ipynb
2. Enable GPU (Runtime → Change runtime type → T4)
3. Run cells [1] → [13] in order
```

### Option B: Create your own notebook

```
1. Open Google Colab
2. Follow steps in docs/COLAB_SETUP.md
3. Copy code cells as needed
```

## What You'll Train

**Model:** YOLOv8-Medium (fine-tuned for PPE detection)

**Classes:** 10 safety equipment classes
- Worker, Drone
- Safety glasses, Safety goggles
- Hard hat, Regular hat
- Hi-vis vest, Regular clothing
- Work boots, Regular shoes

**Dataset:** ~1.4K-19K images (depending on optional datasets)
- Primary: Ultralytics Construction-PPE (1.4K images, auto-download)
- Optional: Roboflow (2.8K), Hard Hat Universe (7K), SH17 (8K)

**Performance:** Expected mAP50 = 0.85+ on validation set

## Expected Results

After training (30 epochs on T4 GPU):

✅ **Trained model** (`ppe_detector.pt`)
✅ **ONNX export** (`ppe_detector.onnx`) for edge deployment
✅ **Validation metrics** (mAP, precision, recall per class)
✅ **Sample inferences** showing PPE detections
✅ **Training logs** with plots and curves

**Location:** `/content/drive/MyDrive/YOLOv8_PPE_Training/`

## Training Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup (Mount Drive, install deps) | 5 min | Automatic |
| Dataset download | 20 min | Ultralytics auto-download |
| Validation checks | 2 min | GPU + dataset verification |
| Fine-tuning training | 2-3 hrs | 30 epochs on T4 GPU |
| Export to ONNX | 5 min | For edge deployment |
| Save to Drive | 2 min | Automatic backup |
| **Total** | **~3 hours** | On T4 GPU |

## Important Notes

### Security
- **Never commit personal Colab URLs** to the repository
- **Keep API keys private** (Roboflow, Kaggle)
- Use throwaway test accounts for optional data sources

### GPU Tiers
- **T4 GPU** (free): ~40 min per epoch, good for testing
- **A100 GPU** (paid): ~10 min per epoch, production training
- **CPU only** (not recommended): 5+ hours per epoch

### Customization
Each section of the notebook is self-contained and can be modified:
- Change batch size (line in cell [10])
- Adjust learning rate (line in cell [10])
- Use different dataset (modify cell [5-8])
- Change output paths (modify cell [12])

## Next Steps After Training

1. **Download models** from Google Drive
2. **Test locally** on warehouse video footage
3. **Validate performance** using validation metrics
4. **Deploy to edge** by converting to Qualcomm QNN DLC format

See `docs/DEPLOYMENT.md` for deployment instructions.

## Troubleshooting

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in cell [10] from 16 to 8 or 4

### Problem: "Dataset not downloading"
**Solution:** Check cell [5] output, ensure Ultralytics auto-download triggered

### Problem: "Models not saving to Drive"
**Solution:** Verify Drive mount (cell [1]) completed successfully

For more troubleshooting: See `COLAB_QUICKSTART.md` or `docs/COLAB_SETUP.md`

## References

- **Ultralytics Docs:** https://docs.ultralytics.com
- **Google Colab Help:** https://colab.research.google.com/notebooks/basic_features_overview.ipynb
- **ONNX Runtime:** https://onnxruntime.ai/
- **Qualcomm QNN:** https://docs.qualcomm.com/bundle/qnn-sdk

---

**Last Updated:** April 2026  
**Status:** Production-ready for Google Colab T4 GPU training
