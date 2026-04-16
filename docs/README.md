# Yolov8 Powered Safety Equipment Detection System

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/a-kuo/yolov8-powered-safety-equipment-detection-system.git
cd yolov8-powered-safety-equipment-detection-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Inference (Demo)
```bash
python scripts/run_inference_video.py --video sample.mp4 --model worker_detector
```

### 3. Test Setup
```bash
pytest tests/ -v
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System design and dataflow
- **[SETUP.md](SETUP.md)** — Detailed environment setup
- **[MODEL_TRAINING.md](MODEL_TRAINING.md)** — Training and fine-tuning
- **[DEPLOYMENT.md](DEPLOYMENT.md)** — Deploy to edge devices
- **[QNN_OPTIMIZATION.md](QNN_OPTIMIZATION.md)** — ONNX → QNN workflow
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — Common issues

## Features

✅ Real-time worker detection using YOLOv8  
✅ Fine-grained PPE compliance checking (glasses, hats, vests, shoes)  
✅ Multi-backend support (PyTorch, ONNX, Qualcomm QNN)  
✅ Edge optimization for Qualcomm devices  
✅ Telemetry and logging  

## System Requirements

- **GPU**: NVIDIA CUDA 12.1+ OR Intel Arc OR Qualcomm Snapdragon
- **CPU**: Intel Core Ultra 7+ or equivalent
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 10GB for models and datasets

## Model Architecture

```
Input Frame (640×640)
    ↓
Worker Detector (YOLOv8-N)
    ↓
Drone Classifier (YOLOv8-Cls)
    ↓
PPE Detector (YOLOv8-M)
    ↓
Safety Compliance Report
```

## Performance

- **FPS**: 30 FPS on edge devices
- **Latency**: ~33ms per frame
- **Memory**: <2GB on Snapdragon
- **Accuracy**: 95%+ worker detection, 90%+ PPE

## Configuration

Edit `config/models.yaml` and `config/dataset.yaml` to customize:
- Confidence thresholds
- Model architectures
- Class definitions
- Training hyperparameters

## License

See LICENSE file for details.

## Support

For issues or questions, open an issue on GitHub.
