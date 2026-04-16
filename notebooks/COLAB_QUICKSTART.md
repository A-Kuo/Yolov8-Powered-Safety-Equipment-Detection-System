# Google Colab Training — Quick Start Guide

## 🚀 Fast Track (15 minutes to training)

### Step 1: Open Colab
Click this link to open the notebook:
> **https://colab.research.google.com/github/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System/blob/claude/setup-yolo-safety-detection-UoZTY/notebooks/Colab_Training_PPE_Detection.ipynb**

Or:
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "Open notebook" → "GitHub"
3. Enter: `A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System`
4. Select `notebooks/Colab_Training_PPE_Detection.ipynb`

### Step 2: Enable GPU
1. Click "Runtime" menu
2. Select "Change runtime type"
3. Choose: **GPU** (T4 or A100)
4. Click "Save"

### Step 3: Run Cells in Order
- **[1-3]:** Setup & dependencies (10 min)
- **[4]:** Download Ultralytics dataset (5 min, zero setup)
- **[5-6]:** Optional Roboflow/Kaggle (skip if no API keys)
- **[7-8]:** Organize datasets (2 min)
- **[9-10]:** Merge datasets (5 min)
- **[11]:** Download pretrained model (5 min)
- **[12]:** Start training (2-3 hours) ⏳
- **[13-16]:** Export & save results (10 min)

---

## 📊 Training Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Setup & Dependencies | 10 min | Automatic pip install |
| Dataset Download | 30 min | Ultralytics (free), Roboflow (optional) |
| Dataset Merge | 5 min | Automatic class mapping |
| Fine-tuning (30 epochs) | 2-3 hours | GPU accelerated |
| Validation | 10 min | mAP, precision, recall |
| Export (ONNX + TorchScript) | 5 min | Ready for deployment |
| **TOTAL** | **~4 hours** | 2-3x faster than local CPU |

---

## 🔑 API Keys (Optional)

### Roboflow (Optional)
Get more training data (7K+ helmet images, 2.8K construction images):

1. Sign up (free): https://roboflow.com
2. Get API key: https://app.roboflow.com/settings/api
3. In cell **[5]**, set: `ROBOFLOW_API_KEY = "your_key_here"`

### Kaggle (Optional)
Download SH17 dataset (8K images, 17 PPE classes):

1. Install Kaggle CLI: `pip install kaggle`
2. Download credentials from: https://www.kaggle.com/settings/account
3. Upload `kaggle.json` to Colab via **Files** panel
4. Run cell **[6]**

---

## 💾 Where Do Results Go?

All results automatically save to **Google Drive**:

```
My Drive/YOLOv8_PPE_Training/
├── models/
│   ├── ppe_detector.pt (PyTorch)
│   ├── ppe_detector.onnx (ONNX Runtime)
│   └── ppe_detector.torchscript (TorchScript)
├── training_results/
│   ├── weights/best.pt
│   ├── results.csv
│   └── plots/
├── inference_samples.png
├── TRAINING_SUMMARY.txt
└── DEPLOYMENT_GUIDE.md
```

Download these files to test locally or deploy to Snapdragon.

---

## 🎯 What You'll Get

### 1. **Trained Model** (ppe_detector.pt)
Fine-tuned on 1.4K-19K images (depending on datasets):
- **Detects:** Workers, helmets, hi-vis vests, goggles, glasses, boots, shoes
- **Performance:** mAP50 = 0.85+ expected
- **Speed:** 30 FPS on GPU, 5-15 FPS on CPU

### 2. **ONNX Model** (ppe_detector.onnx)
Cross-platform inference:
- Works on any CPU/GPU via ONNX Runtime
- Can be converted to Qualcomm QNN .dlc for Snapdragon

### 3. **Validation Report**
- mAP@50, mAP@50-95 (how accurate?)
- Precision, Recall (false positives vs false negatives)
- Per-class performance (which classes are hardest?)

### 4. **Sample Inferences**
- 3 test images with PPE detections highlighted
- Proof that the model works

---

## ⚠️ Troubleshooting

### "CUDA out of memory"
→ In cell **[11]**, reduce `batch=16` to `batch=8` or `batch=4`

### "Dataset not found"
→ Make sure cell **[4]** or **[5]** completed successfully
→ Check that `/content/workspace/data/raw/` has directories

### "No modules named 'roboflow'"
→ Cell **[2]** installs it automatically, but pip errors can happen
→ Manually run: `!pip install -q roboflow`

### "Training stuck / taking forever"
→ Check that Runtime has GPU enabled (Runtime → Change runtime type)
→ Check GPU availability: Cell outputs should show `CUDA available: True`

### "Models saved but can't download"
→ Check Google Drive at: Drive → My Drive → YOLOv8_PPE_Training
→ If not there, manually save: Cell **[15]** has the save code

---

## 🔄 Iterate & Improve

### After first training (30 epochs):
1. Download results from Drive
2. Test on local warehouse video
3. If accuracy is low: Re-run with more epochs
4. If overfitting (val loss > train loss): Add more Roboflow data

### Command to extend training:
```python
# In a new cell, resume from checkpoint
model = YOLO('/content/workspace/runs/ppe_finetune_v1/weights/last.pt')
results = model.train(
    data='/content/workspace/data/merged_dataset.yaml',
    epochs=60,  # More epochs
    patience=15,
    device=0,
    ...
)
```

---

## 📤 Deploy to Snapdragon

### After training:

1. **Get ONNX model** from Drive
2. **Convert to QNN DLC** (Qualcomm AI Hub):
   ```bash
   # Via Qualcomm AI Hub web interface
   # Upload ppe_detector.onnx
   # Download ppe_detector.dlc
   ```

3. **Run on Snapdragon/Rubik Pi:**
   ```python
   from src.edge_deployment.qnn_executor import QNNExecutor
   
   executor = QNNExecutor('ppe_detector.dlc')
   for frame in video_stream:
       results = executor.infer(frame)
       # Check PPE compliance...
   ```

---

## 📚 References

- **Colab Notebook:** `notebooks/Colab_Training_PPE_Detection.ipynb`
- **Dataset Guide:** `docs/DATA_SOURCES.md`
- **Data Merging:** `data/preprocessing/merge_datasets.py`
- **Class Mapping:** `data/annotations/class_mapping.yaml`
- **Deployment:** `docs/DEPLOYMENT.md`

---

## 🎓 What Happens Inside

### Training Pipeline

```
1. Load pretrained model (keremberke YOLOv8m)
   └─ Already knows what helmets, goggles, gloves look like

2. Freeze backbone layers (transfer learning)
   └─ Reuse learned features, fine-tune head

3. Train on merged dataset
   ├─ 1,416 Ultralytics images (helmets, vests, boots, goggles)
   ├─ 2,801 Roboflow images (vest + helmet compliance)
   ├─ 7,036 Hard Hat Universe images (helmet boost)
   └─ 8,099 SH17 images (gloves, shoes, diversity)
   └─ Total: ~19K images (if all datasets downloaded)

4. Validate on test split
   └─ Measure mAP, precision, recall per class

5. Export to ONNX
   └─ Compatible with any framework (TensorFlow, CoreML, QNN)
```

### Class Distribution
```
High-confidence detections (easy):
  - Workers (person): 20,000+ samples
  - Hard hats: 14,000+ samples
  - Hi-vis vests: 3,800+ samples

Medium-confidence (moderate):
  - Safety goggles: 3,500 samples
  - Work boots: 2,000 samples

Low-confidence (hard):
  - Safety glasses: 500 samples ⚠️ may need more data
  - Gloves: 1,800 samples ⚠️ borderline
  - Drones: ~0 samples ❌ will need synthetic data
```

---

## 🎉 When Training Is Done

You'll have:

✅ **Trained model** ready for production  
✅ **Validation metrics** showing real performance  
✅ **Inference examples** proving it works  
✅ **ONNX export** for cross-platform deployment  
✅ **Full reproducibility** — all code in repo  

**Next:** Test on real warehouse video, deploy to Snapdragon, iterate!

---

**Questions?** Check:
- `docs/DATA_SOURCES.md` — where data comes from
- `docs/ARCHITECTURE.md` — how the system works
- `CLAUDE.md` — codebase overview
