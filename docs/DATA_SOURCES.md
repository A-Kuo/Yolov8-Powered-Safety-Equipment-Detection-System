# PPE Detection Data Sources & Strategy

## Overview

This document covers all datasets, pretrained models, and data acquisition strategy for the YOLOv8 PPE detection system. Datasets are prioritized by the classes that are **most detectable with YOLOv8** and most critical for warehouse safety compliance.

**Detection Priority:**
1. 🟡 **Hi-Vis Vest** — Largest surface area, high contrast, easiest to detect
2. 🟡 **Hard Hat / Helmet** — Distinctive shape on head, well-studied
3. 🟠 **Safety Goggles / Glasses** — Moderate difficulty (smaller region)
4. 🔴 **Gloves** — Hard (small, many colors, often occluded)
5. 🔴 **Steel Toe Boots** — Hardest (bottom of frame, occluded by ground)

---

## Primary Datasets (Recommended)

### 1. Roboflow Construction Site Safety ⭐⭐⭐⭐⭐
**Best starting dataset — hi-vis vest + helmet focus**

| Property | Value |
|----------|-------|
| Source | Roboflow Universe |
| ID | `roboflow-universe-projects/construction-site-safety` |
| Images | ~2,801 (v27 YOLOv8s) |
| Format | YOLOv8 TXT |
| License | CC BY 4.0 |
| Classes | 10 (see below) |

**Classes:**
```
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, 
Person, Safety Cone, Safety Vest, machinery, vehicle
```

**Why use it:** Directly annotates both compliant (`Safety Vest`, `Hardhat`) and non-compliant (`NO-Safety Vest`, `NO-Hardhat`) states. Critical for safety alerting.

**Download:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
dataset = project.version(27).download("yolov8")
```

---

### 2. Hard Hat Universe ⭐⭐⭐⭐⭐
**Largest dedicated helmet dataset**

| Property | Value |
|----------|-------|
| Source | Roboflow Universe |
| ID | `universe-datasets/hard-hat-universe-0dy7t` |
| Images | ~7,036 |
| Format | YOLOv8 TXT |
| License | CC BY 4.0 |
| Classes | 3 (helmet, head, person) |

**Why use it:** 7K+ images dedicated to helmet detection. Massive class imbalance corrector for hard hat class.

**Download:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("universe-datasets").project("hard-hat-universe-0dy7t")
dataset = project.version(1).download("yolov8")
```

---

### 3. Ultralytics Construction-PPE ⭐⭐⭐⭐
**Built into Ultralytics — zero setup needed**

| Property | Value |
|----------|-------|
| Source | Ultralytics (built-in) |
| Download | Auto via `data=construction-ppe.yaml` |
| Images | 1,416 total (1,132 train / 143 val / 141 test) |
| Format | YOLOv8 YAML (auto-managed) |
| License | AGPL-3.0 |
| Size | 178.4 MB |

**Classes (11 total):**
```
helmet, gloves, vest, boots, goggles, none,
Person, no_helmet, no_goggle, no_gloves, no_boots
```

**Why use it:** Only dataset with `boots` and `goggles` annotations. Zero download friction — just pass the YAML. Covers the full PPE compliance picture.

**Usage:**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
```

---

### 4. SH17 Dataset ⭐⭐⭐⭐
**Best academic dataset — most diverse, 17 classes**

| Property | Value |
|----------|-------|
| Source | Kaggle / Zenodo / GitHub |
| Kaggle ID | `mugheesahmad/sh17-dataset-for-ppe-detection` |
| Images | 8,099 annotated |
| Instances | 75,994 across 17 classes |
| Format | YOLOv8 TXT |
| License | CC BY-NC-SA 4.0 (non-commercial) |

**Classes (17 total):**
```
Person, Head, Face, Glasses, Face-mask-medical, Face-guard,
Ear, Earmuffs, Hands, Gloves, Foot, Shoes, Safety-vest,
Tools, Helmet, Medical-suit, Safety-suit
```

**Why use it:** Only dataset covering `Gloves`, `Shoes/Boots`, `Safety-vest`, and `Helmet` in the same images. 8K diverse industrial images.

**Download:**
```bash
# Via Kaggle CLI
pip install kaggle
kaggle datasets download mugheesahmad/sh17-dataset-for-ppe-detection
unzip sh17-dataset-for-ppe-detection.zip -d data/raw/sh17/

# Via GitHub
git clone https://github.com/ahmadmughees/SH17dataset data/raw/sh17/
```

---

## Secondary Datasets

### 5. Hard Hat Workers (Joseph Nelson / Roboflow)
| Property | Value |
|----------|-------|
| Source | Roboflow Universe |
| ID | `joseph-nelson/hard-hat-workers` |
| Images | ~5,269 |
| Classes | helmet, head, person |
| License | Public |

Classic dataset, used in many benchmark papers.

---

### 6. PPE Dataset for Workplace Safety
| Property | Value |
|----------|-------|
| Source | Roboflow Universe |
| ID | `ppe-la0vn/ppe-dataset-for-workplace-safety-qobrx` |
| Images | ~3,000+ |
| Classes | PPE, No-PPE compliance pairs |
| License | CC BY 4.0 |

---

## Pretrained Models

### 1. keremberke/yolov8m-protective-equipment-detection ⭐⭐⭐⭐⭐
**Best ready-to-use PPE model — start here for inference**

| Property | Value |
|----------|-------|
| Source | Hugging Face |
| Model | YOLOv8-Medium |
| Training Images | 6,473 train / 3,570 val / 1,935 test |
| License | MIT |

**Classes:**
```
glove, goggles, helmet, mask,
no_glove, no_goggles, no_helmet, no_mask,
no_shoes, shoes
```

**Download:**
```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
    repo_id="keremberke/yolov8m-protective-equipment-detection",
    filename="best.pt"
)
model = YOLO(model_path)
```

**Also available:** `yolov8s-` and `yolov8n-` variants for lighter deployment.

---

### 2. VoxDroid Construction Safety (GitHub)
**Pretrained on Roboflow construction dataset, highest reported mAP**

| Property | Value |
|----------|-------|
| Source | GitHub |
| Repo | `VoxDroid/Construction-Site-Safety-PPE-Detection` |
| Model | YOLOv8s |
| mAP@50 | 0.877 |
| Precision | 0.95 |
| Recall | 0.798 |

**Classes:** Hardhat, Mask, Safety Vest, Safety Cone, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Vehicle, Machinery

---

### 3. YOLOv8n COCO (Ultralytics baseline — human detection)
**Use for the Worker Detector stage of the pipeline**

| Property | Value |
|----------|-------|
| Source | Ultralytics |
| Classes | 80 COCO classes including `person` |
| mAP@50 | 52.9% (person class) |
| Model Size | 6.2 MB |

**Download:**
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Auto-downloads, person class = 0
```

---

## Data Acquisition Strategy

### Phase 1: Bootstrap (0 custom data)
Use pretrained models directly for early testing:
```
keremberke/yolov8m-protective-equipment-detection
→ Fine-tune on construction-ppe.yaml
```

### Phase 2: Combined Training Dataset
Merge these sources for full PPE coverage:

| Dataset | Images | Key Classes Added |
|---------|--------|-------------------|
| Ultralytics Construction-PPE | 1,416 | boots, goggles (built-in) |
| Roboflow Construction Safety | 2,801 | hi-vis vest, helmet, NO-vest |
| Hard Hat Universe | 7,036 | helmet volume boost |
| SH17 | 8,099 | gloves, shoes diversity |
| **Total** | **~19,352** | **Full PPE coverage** |

### Phase 3: Augmentation
Use `data/preprocessing/augmentation.py` to expand dataset 3-5x:
- Horizontal flip
- Brightness/contrast jitter (important for hi-vis vests in shadows)
- Random scale (for workers at different distances)
- Mosaic (YOLOv8 built-in)
- Perspective transform (different camera angles)

---

## Class Mapping (Source → Target)

Our project uses 10 target classes. This table maps source dataset labels to our schema:

| Target Class | Source Mappings |
|---|---|
| `worker` | `Person` (all datasets) |
| `drone` | No existing public dataset — synthesize |
| `safety_glasses` | `Glasses` (SH17) |
| `safety_goggles` | `goggles` (keremberke), `no_goggles` → neg |
| `hard_hat` | `Hardhat`, `helmet`, `Helmet` (all datasets) |
| `regular_hat` | `Head` (Hard Hat Universe, non-helmet) |
| `hi_vis_vest` | `Safety Vest`, `vest`, `Safety-vest` |
| `regular_clothing` | `NO-Safety Vest` (negative class) |
| `work_boots` | `Shoes` (SH17), `boots` (Construction-PPE) |
| `regular_shoes` | `no_shoes` (keremberke) |

See `data/annotations/class_mapping.yaml` for full mapping config.

---

## Estimated Data Volume for Production

For target accuracy (90%+ mAP):

| Class | Min Images Needed | Available Now |
|-------|-----------------|---------------|
| hi_vis_vest | 2,000 | ~3,800 ✅ |
| hard_hat | 3,000 | ~14,000 ✅ |
| safety_goggles | 1,500 | ~3,500 ✅ |
| safety_glasses | 1,000 | ~500 ⚠️ need more |
| work_boots | 1,500 | ~2,000 ✅ |
| gloves | 2,000 | ~1,800 ⚠️ borderline |
| worker (person) | 5,000 | 20,000+ ✅ |
| drone | 500 | ~0 ❌ need custom data |

---

## API Keys & Accounts Required

| Source | Account Needed | Cost |
|--------|---------------|------|
| Roboflow Universe | Free account | Free (most datasets) |
| Kaggle | Free account | Free |
| Hugging Face | None | Free |
| Ultralytics | None | Free |

Sign up at:
- Roboflow: https://roboflow.com (free tier: 3 projects, 10K images/month)
- Kaggle: https://kaggle.com (requires CLI setup)
- Hugging Face: https://huggingface.co (anonymous download works)

---

## References

- [SH17 Dataset Paper (arxiv)](https://arxiv.org/abs/2407.04590)
- [SH17 Dataset GitHub](https://github.com/ahmadmughees/SH17dataset)
- [SH17 Dataset Kaggle](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)
- [Roboflow Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- [Hard Hat Universe](https://universe.roboflow.com/universe-datasets/hard-hat-universe-0dy7t)
- [keremberke PPE Model (HuggingFace)](https://huggingface.co/keremberke/yolov8m-protective-equipment-detection)
- [VoxDroid Construction Safety (GitHub)](https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection)
- [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
- [Roboflow PPE Browse](https://universe.roboflow.com/browse/construction/ppe)
