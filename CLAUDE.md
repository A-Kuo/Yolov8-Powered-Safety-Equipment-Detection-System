# Codebase Documentation

## Project Overview

**Yolov8 Powered Safety Equipment Detection System** — A real-time worker safety monitoring system using YOLOv8 for detecting human workers and their PPE (Personal Protective Equipment) compliance. Designed for edge deployment on Qualcomm-based hardware (Rubik Pi, Snapdragon) with testing on Intel Arc 140V.

## System Architecture

### Core Detection Pipeline

```
Video Frame
    ↓
Worker Detector (YOLOv8-N)  → Detects humans in warehouse
    ↓
Worker Classifier (YOLOv8-Cls)  → Distinguishes person vs drone
    ↓
PPE Detector (YOLOv8-M)  → Fine-grained PPE classification
    ↓
Safety Alert Generator  → Flags non-compliant workers
    ↓
Telemetry & Logging
```

### Detection Classes

- **Worker Detection**: `worker` (binary human presence)
- **Drone Detection**: `drone` (distinguish from humans)
- **PPE Classes**:
  - `safety_glasses` vs `safety_goggles` (eye protection)
  - `hard_hat` vs `regular_hat` (head protection)
  - `hi_vis_vest` vs `regular_clothing` (visibility)
  - `work_boots` vs `regular_shoes` (foot protection)

## Directory Structure

### `config/`
Configuration files for models, datasets, and inference:
- **models.yaml** — Model definitions, checkpoints, thresholds
- **dataset.yaml** — Class taxonomy, training parameters, data splits
- **inference.yaml** — Runtime parameters, postprocessing, telemetry

### `data/`
Dataset management:
- **raw/** — Original datasets (not tracked in git)
- **processed/** — Preprocessed data splits
- **annotations/** — YOLO format class definitions
- **preprocessing/** — Scripts for data augmentation and validation

### `src/inference/`
Core inference engines:
- **yolo_detector.py** — YOLOv8 wrapper for Ultralytics
- **onnx_runtime.py** — ONNX Runtime with QNN support
- **postprocess.py** — NMS, filtering, confidence scoring
- **worker_classifier.py** — Person vs drone classification
- **ppe_detector.py** — Fine-grained PPE detection

### `src/edge_deployment/`
Edge-specific code for Qualcomm devices:
- **qnn_executor.py** — Qualcomm QNN SDK integration (WIP)
- **camera_handler.py** — Video capture from USB/CSI cameras (WIP)
- **safety_monitor.py** — Main inference loop and alerts (WIP)
- **telemetry.py** — Logging and metrics export (WIP)

### `src/training/`
Model training and fine-tuning:
- **train_worker_detector.py** — YOLOv8 base training
- **train_ppe_detector.py** — Fine-tune for custom PPE classes
- **transfer_learning.py** — Transfer learning utilities

### `src/utils/`
Shared utilities:
- **config_loader.py** — YAML config loading
- **logging.py** — Logging setup
- **conversion.py** — PT → ONNX conversion
- **optimization.py** — ONNX → QNN optimization

### `tests/`
Unit and integration tests:
- **test_inference.py** — Inference engine tests
- **test_worker_detector.py** — Detector validation
- **test_ppe_detector.py** — PPE classification tests
- **test_onnx_export.py** — Model export validation
- **conftest.py** — Pytest fixtures

### `docs/`
Comprehensive documentation:
- **ARCHITECTURE.md** — System design and dataflow
- **SETUP.md** — Environment and dependency setup
- **MODEL_TRAINING.md** — How to train and fine-tune models
- **DEPLOYMENT.md** — Deployment to edge devices
- **QNN_OPTIMIZATION.md** — ONNX to QNN workflow
- **TROUBLESHOOTING.md** — Common issues and solutions

### `scripts/`
Utility scripts:
- **download_models.sh** — Download YOLOv8 base models
- **convert_to_onnx.py** — PT to ONNX conversion
- **optimize_for_qnn.py** — ONNX to QNN compilation
- **run_inference_video.py** — Demo inference on video file
- **benchmark.py** — Performance profiling

## Key Files Reference

### Trained Models ✅
- `models/yolo/ppe_detector.pt` — Trained YOLOv8m model (6.3 MB, PyTorch)
- `models/onnx/ppe_detector.onnx` — ONNX export (13 MB, cross-platform)
- `models/yolo/ppe_detector.torchscript` — TorchScript export (13 MB)
- `TRAINING_RESULTS.md` — Complete training report with metrics

### Configuration
- `config/models.yaml` — Model definitions, confidence thresholds
- `config/dataset.yaml` — Class definitions, training hyperparameters
- `config/inference.yaml` — Runtime settings, postprocessing parameters
- `data/annotations/classes.yaml` — YOLO format class list

### Core Inference
- `src/inference/yolo_detector.py:YOLODetector` — Main detection interface
- `src/inference/onnx_runtime.py:ONNXInference` — ONNX Runtime wrapper
- `src/inference/postprocess.py` — NMS and filtering utilities

### Utilities
- `src/utils/config_loader.py:load_config()` — Load YAML configs
- `src/utils/logging.py:setup_logging()` — Initialize logging

## Development Workflow

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Load Trained Models (Current)
✅ Models are ready in:
- PyTorch: `models/yolo/ppe_detector.pt`
- ONNX: `models/onnx/ppe_detector.onnx`
- TorchScript: `models/yolo/ppe_detector.torchscript`

See `TRAINING_RESULTS.md` for complete metrics.

### 3. Test on Local Warehouse Video
```python
from src.inference.yolo_detector import YOLODetector

detector = YOLODetector('models/yolo/ppe_detector.pt', device='cuda')
results = detector.predict('warehouse_video.mp4')
```

### 4. Run Inference Tests
```bash
pytest tests/test_inference.py -v
pytest tests/test_yolov8_integration.py -v
```

### 5. Fine-tune on Real Data (Next Phase)
```bash
python src/training/train_ppe_detector.py --config config/dataset.yaml --data your_real_data.yaml
```

### 6. Export & Optimize for Qualcomm
```bash
python scripts/convert_to_onnx.py --model models/yolo/ppe_detector.pt
python scripts/optimize_for_qnn.py --onnx models/onnx/ppe_detector.onnx
```

## Backend Support

### PyTorch (Local Testing)
- **Device**: CUDA (Intel Arc, NVIDIA GPU) or CPU
- **Precision**: FP32, FP16
- **Speed**: ~30 FPS on Arc 140V

### ONNX Runtime (Cross-Platform)
- **Device**: CPU, GPU, or Qualcomm QNN (with QNN EP)
- **Precision**: FP32, FP16, quantized INT8
- **Speed**: ~30-60 FPS on Arc 140V

### Qualcomm QNN (Edge Deployment)
- **Target**: Rubik Pi, Snapdragon devices
- **Format**: .dlc (Qualcomm DLC)
- **Precision**: Quantized (INT8 or custom)
- **Speed**: 15-60 FPS depending on device

## Model Specifications

### Worker Detector
- Architecture: YOLOv8-Nano
- Input: 640×640 RGB image
- Output: Bounding boxes + confidence scores
- Purpose: Detect human workers in scenes

### PPE Detector
- Architecture: YOLOv8-Medium
- Input: 640×640 RGB image
- Output: 10 classes (glasses, hats, vests, shoes, etc.)
- Purpose: Fine-grained safety compliance checking

### Drone Classifier
- Architecture: YOLOv8-Nano (classification)
- Input: Cropped region (224×224)
- Output: Person vs Drone probability
- Purpose: Reduce false positives from drones/equipment

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_inference.py::test_yolo_detector -v
```

### With Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## CI/CD Pipelines

### GitHub Actions
- **ci.yml** — Code quality (black, flake8, isort)
- **tests.yml** — Unit tests (pytest)
- **model_validation.yml** — Inference validation on sample images

## Performance Targets

- **FPS**: 30 FPS minimum on edge devices
- **Latency**: 33ms per frame (1000/30)
- **Memory**: <2GB on Snapdragon devices
- **Accuracy**: 95%+ for worker detection, 90%+ for PPE

## Dependencies

See `requirements.txt` for full list. Key packages:
- **ultralytics** — YOLOv8 framework
- **torch** — PyTorch deep learning
- **onnx** — ONNX model format
- **onnxruntime** — ONNX inference engine
- **opencv-python** — Computer vision utilities
- **pyyaml** — Configuration files

## Future Enhancements

1. **Multi-camera Support** — Track workers across multiple viewpoints
2. **Temporal Tracking** — Persistent worker IDs using ByteTrack/SORT
3. **Incident Alerts** — WebSocket notifications for safety violations
4. **Mobile App** — Real-time dashboard on edge device
5. **Model Quantization** — INT8 quantization for faster edge inference
6. **Custom Training** — Fine-tune on customer-specific PPE types

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues:
- Model loading failures
- CUDA/GPU errors
- ONNX export issues
- QNN compilation problems
- Inference speed optimization

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Qualcomm QNN SDK](https://docs.qualcomm.com/bundle/qnn-sdk)
- [Intel Arc GPU Support](https://www.intel.com/content/www/us/en/developer/articles/article/intel-arc-gpu-pytorch-support.html)

## Notes

- Model files (.pt, .onnx, .dlc) are excluded from git via `.gitignore`
- Raw datasets are not tracked; document download sources in `SETUP.md`
- QNN support requires Qualcomm QNN SDK installation on target device
- Testing is done on Intel Arc 140V; final deployment targets Qualcomm devices
