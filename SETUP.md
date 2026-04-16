# Environment Setup Guide

## System Requirements

- **Python**: 3.10 or 3.11
- **OS**: Linux, macOS, or Windows
- **GPU** (optional): NVIDIA CUDA 12.1+, Intel Arc, or Qualcomm Snapdragon
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 10GB for models and datasets

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/a-kuo/yolov8-powered-safety-equipment-detection-system.git
cd yolov8-powered-safety-equipment-detection-system
```

### 2. Create Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Test imports
python -c "from src.inference import YOLODetector; print('✓ Setup successful')"

# Run tests
pytest tests/ -v
```

## Configuration

### Environment Variables
Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
# Edit .env with your settings
```

Key variables:
- `DEVICE` — cuda, cpu, or mps (Apple Silicon)
- `CONFIDENCE_THRESHOLD` — Detection confidence (0.0-1.0)
- `LOG_LEVEL` — DEBUG, INFO, WARNING, or ERROR

## GPU Setup (Optional)

### NVIDIA CUDA
```bash
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Intel Arc
```bash
# Intel Arc is supported via standard PyTorch CUDA backend
python -c "import torch; print(torch.cuda.is_available())"
```

### Qualcomm QNN (Edge Devices Only)
On Rubik Pi or Snapdragon:
```bash
pip install onnxruntime-qnn
```

## Download Base Models (Optional)

Download pre-trained YOLOv8 models:
```bash
bash scripts/download_models.sh
```

This downloads:
- YOLOv8-Nano (worker detection)
- YOLOv8-Medium (PPE detection)

Models will be saved to `models/yolo/` and excluded from git.

## Project Structure

```
├── config/               # Configuration files (YAML)
├── data/                 # Datasets (excluded from git)
├── src/
│   ├── inference/       # Core detection engines
│   ├── edge_deployment/ # Edge device code
│   ├── training/        # Model training
│   └── utils/           # Shared utilities
├── tests/               # Test suite (pytest)
├── scripts/             # Utility scripts
├── docs/                # Documentation
├── CLAUDE.md            # Codebase overview
├── requirements.txt     # Python dependencies
└── pyproject.toml       # Project metadata
```

## Running Inference

### Inference on Image
```python
from src.inference import YOLODetector
import cv2

detector = YOLODetector("models/yolo/worker_detector.pt", device="cuda")
image = cv2.imread("test.jpg")
results = detector.predict(image)

print(f"Found {len(results['boxes'])} workers")
```

### Inference on Video
```bash
python scripts/run_inference_video.py --video sample.mp4 --model worker_detector
```

## Training Models

Fine-tune on custom dataset:
```bash
python src/training/train_ppe_detector.py \
    --config config/dataset.yaml \
    --data path/to/dataset \
    --epochs 100
```

## Converting to ONNX

Export PyTorch models to ONNX:
```bash
python scripts/convert_to_onnx.py \
    --model models/yolo/ppe_detector.pt \
    --output models/onnx/ppe_detector.onnx
```

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Test
```bash
pytest tests/test_inference.py::TestYOLODetector -v
```

### With Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### CUDA Not Available
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Model Not Found
```bash
# Download models
bash scripts/download_models.sh

# Or download manually from Ultralytics hub
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Out of Memory
```bash
# Reduce batch size in config/inference.yaml
# Or use CPU instead of GPU
export DEVICE=cpu
```

### ONNX Runtime Issues
```bash
# Reinstall ONNX Runtime
pip uninstall onnxruntime -y
pip install onnxruntime
```

## Next Steps

1. **Prepare Dataset** — Acquire warehouse worker safety images
2. **Fine-tune Models** — Train on custom PPE classes
3. **Test Inference** — Validate on sample videos
4. **Deploy to Edge** — Export to ONNX/QNN and deploy to Snapdragon
5. **Monitor Performance** — Track FPS and accuracy metrics

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime Docs](https://onnxruntime.ai/)
- [Qualcomm QNN SDK](https://docs.qualcomm.com/bundle/qnn-sdk)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Support

For issues or questions:
1. Check `docs/TROUBLESHOOTING.md`
2. Review `CLAUDE.md` for codebase overview
3. Open an issue on GitHub
