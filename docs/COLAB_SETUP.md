# Google Colab Training Setup Guide

This guide shows how to set up and run YOLOv8 PPE safety detection training on Google Colab.

## Quick Start

### Step 1: Create a New Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **New notebook** or upload the provided template
3. Make sure you're logged into a Google account with Drive access

### Step 2: Set Up GPU

In the first cell, set up GPU acceleration:

```python
# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

To enable GPU: **Runtime** → **Change runtime type** → Set **Hardware accelerator** to **T4 GPU** or **A100**

### Step 3: Mount Google Drive

Mount your Google Drive to save trained models:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Clone Repository

Clone the project repository:

```python
import subprocess
result = subprocess.run(
    ['git', 'clone', '--depth=1',
     'https://github.com/A-Kuo/Yolov8-Powered-Safety-Equipment-Detection-System.git',
     '/content/project'],
    capture_output=True,
    text=True
)
print("Repository cloned" if result.returncode == 0 else f"Error: {result.stderr}")
```

### Step 5: Install Dependencies

```python
import subprocess
import sys

deps = [
    'ultralytics==8.1.0',
    'onnxruntime==1.17.0',
    'albumentations==1.3.1',
    'roboflow>=1.1.0',
    'huggingface-hub>=0.20.0'
]

result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q'] + deps,
    capture_output=True,
    text=True
)
print("Dependencies installed" if result.returncode == 0 else f"Warning: {result.stderr[:200]}")
```

### Step 6: Download Dataset

Option A: Use Ultralytics built-in Construction-PPE dataset (fastest):

```python
from ultralytics import YOLO
import os

# Create dataset directory
os.makedirs('/content/workspace/data/processed/merged/train/images', exist_ok=True)

# Download via ultralytics (auto-downloads on first training attempt)
print("Dataset will auto-download on first training run")
```

Option B: Download from Roboflow (requires API key):

```python
from roboflow import Roboflow

# Get API key from https://roboflow.com (free account)
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("construction-site-safety")
dataset = project.download("yolov8")
```

### Step 7: Create Dataset Configuration

```python
import yaml

dataset_yaml = {
    'path': '/content/workspace/data/processed/merged',
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 10,
    'names': {
        0: 'worker',
        1: 'drone',
        2: 'safety_glasses',
        3: 'safety_goggles',
        4: 'hard_hat',
        5: 'regular_hat',
        6: 'hi_vis_vest',
        7: 'regular_clothing',
        8: 'work_boots',
        9: 'regular_shoes'
    }
}

with open('/content/workspace/dataset.yaml', 'w') as f:
    yaml.dump(dataset_yaml, f)
print("Dataset YAML created")
```

### Step 8: Start Training

```python
from ultralytics import YOLO
import torch

# Load pretrained model
model = YOLO('yolov8m.pt')

# Fine-tune on your dataset
results = model.train(
    data='/content/workspace/dataset.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    device=0 if torch.cuda.is_available() else 'cpu',
    patience=10,
    lr0=0.001,  # Fine-tuning learning rate
    freeze=10,   # Transfer learning: freeze first 10 layers
    amp=True,    # Mixed precision
    save=True,
    project='/content/workspace/runs',
    name='ppe_v1',
)
print("Training complete!")
```

### Step 9: Export Models

```python
from ultralytics import YOLO
import os

# Load best model
best_model = YOLO('/content/workspace/runs/ppe_v1/weights/best.pt')

# Export to ONNX
os.makedirs('/content/workspace/models/exports', exist_ok=True)
onnx_path = best_model.export(format='onnx', imgsz=640, opset=12)
print(f"ONNX model: {onnx_path}")

# Save PyTorch model
import shutil
shutil.copy(
    '/content/workspace/runs/ppe_v1/weights/best.pt',
    '/content/workspace/models/exports/ppe_detector.pt'
)
print("PyTorch model saved")
```

### Step 10: Save to Google Drive

```python
import shutil
import os

drive_path = '/content/drive/MyDrive/YOLOv8_PPE_Training'
os.makedirs(drive_path, exist_ok=True)

shutil.copytree(
    '/content/workspace/runs/ppe_v1',
    f'{drive_path}/ppe_v1',
    dirs_exist_ok=True
)
print(f"Saved to: {drive_path}")
```

## Full Training Notebook Template

We provide a complete, ready-to-use notebook at:
```
notebooks/Colab_Training_PPE_Detection.ipynb
```

This notebook includes:
- ✓ Automatic GPU detection and setup
- ✓ Error handling with PASS/FAIL status reporting
- ✓ Dataset auto-download from Ultralytics
- ✓ 30-epoch fine-tuning with transfer learning
- ✓ ONNX export for edge deployment
- ✓ Automatic Google Drive backup

## Creating Your Own Notebook

To create a custom notebook:

1. Open https://colab.research.google.com
2. Create a new Python notebook
3. Copy cells from `notebooks/Colab_Training_PPE_Detection.ipynb`
4. Customize paths and hyperparameters as needed

**Important:** Do NOT commit your personal Colab URLs to the repository. Keep notebook sharing URLs private.

## Typical Training Timeline

| Phase | Time | Description |
|-------|------|-------------|
| Setup | 5 min | Mount Drive, install deps |
| Dataset | 20 min | Download Ultralytics or Roboflow data |
| Validation | 2 min | Check GPU and dataset structure |
| Training | 2-3 hrs | 30 epochs on T4 GPU |
| Export | 5 min | Convert to ONNX and save |
| **Total** | **~3 hours** | Full pipeline on T4 GPU |

## Performance Notes

- **T4 GPU**: ~40 min per epoch (30 epochs = 20-25 min total with optimization)
- **A100 GPU**: ~10 min per epoch (much faster, premium tier)
- **CPU Only**: ~5+ hours per epoch (not recommended)

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
batch=8  # instead of 16
```

### Dataset Not Downloading
```python
# Use local dataset instead
# Copy images to /content/workspace/data/processed/merged/train/images
```

### Model Not Exporting
```python
# Ensure model weights exist
import os
assert os.path.exists('/content/workspace/runs/ppe_v1/weights/best.pt')
```

## Next Steps

After training completes:

1. **Download models** from `/content/drive/MyDrive/YOLOv8_PPE_Training/`
2. **Test locally** with your own warehouse video
3. **Validate mAP** on validation dataset
4. **Deploy to Snapdragon** by converting to QNN DLC format

See `docs/DEPLOYMENT.md` for edge device deployment instructions.

## Resources

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [Google Colab Guide](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)
- [Qualcomm QNN SDK](https://docs.qualcomm.com/bundle/qnn-sdk)
