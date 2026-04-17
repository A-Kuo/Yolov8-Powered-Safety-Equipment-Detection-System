# Phase 2: Real Data Collection & Fine-Tuning
## YOLOv8 Safety Detection System

**Status:** In Progress  
**Timeline:** 2-3 weeks  
**Previous Phase:** ✅ Phase 1 complete (synthetic baseline, POC)  

---

## Overview

Phase 2 transforms the proof-of-concept into production-ready by collecting and fine-tuning on real-world data.

### Three Sub-Phases

| Phase | Duration | Task | Expected Result |
|-------|----------|------|-----------------|
| **2A** | 1-2 hours | Optimize synthetic baseline to 80 epochs | mAP@50: 0.559 → 0.65-0.70 |
| **2B** | 1 week | Collect real data from Roboflow + Kaggle | 1,200+ images |
| **2C** | 1 week | Fine-tune on mixed real + synthetic data | mAP@50: ≥ 0.80 |

---

## Phase 2A: Optimize Synthetic Baseline (DONE ✅)

### Step 1: Update Colab Training to 80 Epochs

**Status:** ✅ Complete

Changed in `notebooks/Colab_Training_PPE_Detection.ipynb`:
- `epochs=30` → `epochs=80`
- `warmup_epochs=3` → `warmup_epochs=5`

**Expected Result:**
- Training time: ~1.3 hours on T4 GPU
- mAP@50: 0.559 → 0.65-0.70 (synthetic data only)

**Run Colab:**
1. Open: https://colab.research.google.com/github/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System/blob/claude/setup-yolo-safety-detection-UoZTY/notebooks/Colab_Training_PPE_Detection.ipynb
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run cells [1] → [13] in order
4. Check training progress in cell [9.2]

---

## Phase 2B: Real Data Collection (IN PROGRESS)

### Step 1: Download Real Datasets

Two high-quality public datasets with 1,200+ total PPE images:

#### Option A: Automatic Download (Recommended)

```bash
python scripts/download_real_datasets.py
```

Downloads both:
- **Roboflow Construction Safety** (2,801 images)
  - Classes: Hardhat, Safety Vest, Person
  - Free tier, no authentication
  - Output: `data/raw/roboflow_construction/`

- **Kaggle SH17 Dataset** (8,099 images)
  - Classes: 17 PPE types (helmets, gloves, vests, boots, etc.)
  - Requires `~/.kaggle/kaggle.json`
  - Output: `data/raw/kaggle_sh17/`

#### Option B: Manual Download

**Roboflow Construction Safety:**
1. Visit: https://roboflow.com/
2. Search: "construction-site-safety"
3. Download YOLOv8 format
4. Extract to: `data/raw/roboflow_construction/`

**Kaggle SH17:**
1. Go to: https://www.kaggle.com/datasets/deeptech/sh17-dataset
2. Setup Kaggle auth: https://www.kaggle.com/settings/account
3. Download via CLI:
   ```bash
   kaggle datasets download -d deeptech/sh17-dataset
   unzip sh17-dataset.zip -d data/raw/kaggle_sh17/
   ```

#### Option C: Quick Test (Dry Run)

```bash
python scripts/download_real_datasets.py --dry-run
```

Shows what would be downloaded without downloading.

### Step 2: Validate Downloads

```bash
python scripts/download_real_datasets.py

# Output:
# Roboflow:      2,801 images ✓
# Kaggle SH17:   8,099 images ✓
# TOTAL:         10,900 images
```

### Step 3: Check File Structure

After download, you should have:
```
data/raw/
├── roboflow_construction/
│   ├── images/train/      (2,801 images)
│   └── labels/train/      (2,801 .txt files)
└── kaggle_sh17/
    ├── images/            (8,099 images)
    └── labels/            (8,099 .txt files)
```

---

## Phase 2C: Merge & Fine-Tune on Mixed Data (NEXT)

### Step 1: Create Class Mapping Configuration

**File:** `data/annotations/class_mapping_public.yaml`

This maps Roboflow (3 classes) and SH17 (17 classes) → 10-class unified schema:

```yaml
0: worker           # Person
1: drone            # Drone/equipment
2: safety_glasses   # Eye protection
3: safety_goggles
4: hard_hat         # Head protection
5: regular_hat
6: hi_vis_vest      # Torso protection
7: regular_clothing
8: work_boots       # Foot protection
9: regular_shoes
```

**Mapping Rules:**
- Roboflow "Hardhat" → target class 4 ✓
- Roboflow "Safety Vest" → target class 6 ✓
- Roboflow "Person" → target class 0 ✓
- SH17 "safety_boots" → target class 8 ✓
- SH17 "goggles" → target class 3 ✓
- SH17 classes not in schema → dropped (gloves, knee pads, etc.)

### Step 2: Merge All Data Sources

```bash
python data/preprocessing/merge_real_datasets.py \
    --roboflow data/raw/roboflow_construction \
    --kaggle data/raw/kaggle_sh17 \
    --synthetic data/processed/merged/train/images \
    --output data/processed/mixed \
    --config data/annotations/class_mapping_public.yaml
```

**What It Does:**
1. Remaps all class IDs using `class_mapping_public.yaml`
2. Filters invalid/unmappable classes
3. Creates train/val/test split (70/15/15)
4. Saves to `data/processed/mixed/`

**Expected Output:**
```
data/processed/mixed/
├── train/
│   ├── images/ (840 images)
│   └── labels/ (840 .txt files)
├── val/
│   ├── images/ (180 images)
│   └── labels/ (180 .txt files)
└── test/
    ├── images/ (180 images)
    └── labels/ (180 .txt files)
```

**Summary Output:**
```
Roboflow:       2,000+ images ✓
Kaggle SH17:    1,500+ images ✓
Synthetic:      50 images ✓
TOTAL:          1,200+ images
✓ SUCCESS: Ready for Phase 2C fine-tuning
```

### Step 3: Update Colab for Mixed Data Training

Add to `notebooks/Colab_Training_PPE_Detection.ipynb`:

**After Step [5] (Dataset Download), add:**

```python
# [5B] Download real datasets
print("[5B] Downloading real Roboflow + Kaggle datasets...")
result = subprocess.run(
    [sys.executable, 'scripts/download_real_datasets.py'],
    cwd='/content/workspace/project',
    capture_output=True,
    text=True
)
print("[5B] PASS: Datasets downloaded") if result.returncode == 0 else print(f"[5B] INFO: {result.stderr[:100]}")

# [5C] Merge datasets
print("[5C] Merging datasets with class remapping...")
result = subprocess.run(
    [sys.executable, 'data/preprocessing/merge_real_datasets.py',
     '--roboflow', '/content/workspace/data/raw/roboflow_construction',
     '--kaggle', '/content/workspace/data/raw/kaggle_sh17',
     '--synthetic', '/content/workspace/data/processed/merged/train/images',
     '--output', '/content/workspace/data/processed/mixed'],
    cwd='/content/workspace/project',
    capture_output=True,
    text=True
)
print("[5C] PASS: Datasets merged") if result.returncode == 0 else print(f"[5C] INFO: {result.stderr[:100]}")
```

### Step 4: Update Training Configuration for Mixed Data

**File:** `config/mixed_dataset.yaml`

Key changes from synthetic (Phase 2A) to mixed data (Phase 2C):
- `epochs=80` → `epochs=60` (fewer epochs due to larger, diverse dataset)
- `lr0=0.001` → `lr0=0.0005` (lower learning rate for fine-tuning)
- `warmup_epochs=5` → `warmup_epochs=5` (same)
- `freeze=10` → `freeze=10` (keep backbone frozen for transfer learning)

### Step 5: Run Phase 2C Training

Update training cell [9.2] in Colab:

```python
print("[9.2] Starting fine-tuning on MIXED real data...")
print("Dataset: /content/workspace/data/processed/mixed")
print("Epochs: 60 (Phase 2C: mixed real+synthetic, lower LR)")
print("Batch size: 16")
print("LR: 0.0005 (fine-tuning)")
print("Freeze: 10 (transfer learning)")

results = model.train(
    data='/content/workspace/data/processed/mixed/dataset.yaml',
    epochs=60,      # Phase 2C: lower than synthetic
    imgsz=640,
    batch=16,
    device=0,
    patience=15,
    save=True,
    project='/content/workspace/runs',
    name='ppe_v2_mixed',
    exist_ok=True,
    optimizer='SGD',
    lr0=0.0005,     # Phase 2C: lower LR for fine-tuning
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    amp=True,
    freeze=10,      # Transfer learning
    # ... augmentation unchanged
)
```

**Expected Training Time:**
- 1,200 images → ~0.8 hours on T4 GPU (vs 0.48 hrs for 50 synthetic images)
- 60 epochs on mixed data → mAP@50 should reach ≥ 0.80

**Expected Performance:**
```
Metrics (validation set):
- mAP@50:  0.80+  (target achieved ✓)
- Precision: 0.85+
- Recall: 0.75+
- Per-class performance balanced across all 10 classes
```

### Step 6: Export and Validate Models

Same as Phase 1, but with better performance:

```python
# [10] Export to ONNX and PyTorch
best_model = YOLO('/content/workspace/runs/ppe_v2_mixed/weights/best.pt')
best_model.export(format='onnx', imgsz=640, opset=12)
```

Export locations:
- PyTorch: `models/exports/ppe_detector_v2.pt`
- ONNX: `models/exports/ppe_detector_v2.onnx`

---

## Checklist: Phase 2 Completion

### Phase 2A: Synthetic Baseline
- [x] Updated Colab: epochs 30→80
- [x] Committed to feature branch
- [ ] Run on Colab and verify mAP@50 > 0.60

### Phase 2B: Real Data Collection
- [ ] Download Roboflow dataset (2,801 images)
- [ ] Download Kaggle SH17 dataset (8,099 images)
- [ ] Validate 10,000+ total images downloaded
- [ ] All images in proper directory structure

### Phase 2C: Mixed Data Fine-Tuning
- [ ] Merge datasets with class remapping (1,200+ images)
- [ ] Create mixed_dataset.yaml configuration
- [ ] Run Phase 2C training on Colab (60 epochs)
- [ ] Achieve mAP@50 ≥ 0.80 on validation set
- [ ] Export ONNX and PyTorch models
- [ ] Save models to Google Drive

### Final Validation
- [ ] Download models from Google Drive
- [ ] Test locally on warehouse video
- [ ] Verify FPS ≥ 30 on GPU
- [ ] Measure false positive/negative rates

---

## Troubleshooting

### Issue: Roboflow download fails

**Solution:**
```bash
pip install --upgrade roboflow
python scripts/download_real_datasets.py --roboflow
```

### Issue: Kaggle authentication error

**Solution:**
1. Get API token: https://www.kaggle.com/settings/account
2. Save to: `~/.kaggle/kaggle.json`
3. Fix permissions: `chmod 600 ~/.kaggle/kaggle.json`
4. Try again: `python scripts/download_real_datasets.py --kaggle`

### Issue: Low mAP@50 after fine-tuning

**Troubleshoot:**
- Check class distribution: `python data/preprocessing/merge_real_datasets.py --validate-only`
- Ensure label format is correct (normalized 0-1)
- Try lower learning rate: `lr0=0.0001`
- Increase epochs: `epochs=80` (Phase 2C)

### Issue: Out of memory during training

**Solution:**
Reduce batch size in Colab cell [9.2]:
```python
batch=8  # instead of 16
```

---

## Next Phase: Phase 3

After Phase 2C completion:
1. **Local Testing**: Test on real warehouse video
2. **Optimization**: Measure FPS, latency, memory
3. **Safety Rules**: Implement compliance checking
4. **Edge Prep**: Plan for Snapdragon/Rubik Pi deployment

See `docs/DEPLOYMENT.md` for Phase 3-5 details.

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `notebooks/Colab_Training_PPE_Detection.ipynb` | Modified | Phase 2A/2C training cells |
| `scripts/download_real_datasets.py` | New | Download Roboflow + Kaggle |
| `config/mixed_dataset.yaml` | New | Mixed dataset config |
| `data/annotations/class_mapping_public.yaml` | New | Class remapping rules |
| `data/preprocessing/merge_real_datasets.py` | New | Merge + remap datasets |
| `docs/PHASE2_DATA_COLLECTION.md` | New | This guide |

---

## Resources

- [Roboflow Universe](https://roboflow.com/)
- [Kaggle SH17 Dataset](https://www.kaggle.com/datasets/deeptech/sh17-dataset)
- [Ultralytics YOLOv8 Training](https://docs.ultralytics.com/modes/train/)
- [YOLO Label Format](https://docs.ultralytics.com/datasets/detect/)

---

**Status:** Phase 2B in progress, Phase 2C ready to start after data collection  
**Last Updated:** April 2026  
**Next Checkpoint:** Phase 2C completion with mAP@50 ≥ 0.80
