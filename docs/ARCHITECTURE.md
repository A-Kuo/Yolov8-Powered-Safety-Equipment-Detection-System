# System Architecture

## Overview

The Yolov8 Safety Equipment Detection System is a modular, production-grade edge AI application designed to monitor worker safety in warehouse environments.

## High-Level Design

```
┌─────────────────────────────────────────────────────┐
│         Video Input (Camera/Stream/File)            │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────┐
        │   Frame Preprocessing      │
        │  (Resize, Normalize, etc)  │
        └────────────────┬───────────┘
                         │
                    ┌────┴─────┐
                    │           │
                    ↓           ↓
         ┌──────────────┐   ┌──────────────────┐
         │ Worker Det.  │   │ Alternative:     │
         │ (YOLOv8-N)   │   │ ONNX + QNN       │
         └──────┬───────┘   └──────────────────┘
                │
         ┌──────┴──────┐
         │ No workers? │ → Skip PPE check
         │ (confidence)│
         └──────┬──────┘
                │ Workers found
                ↓
         ┌────────────────────┐
         │ Drone Classifier   │
         │ (YOLOv8-Cls)       │
         │ Filter false +ves  │
         └────────┬───────────┘
                  │
                  ↓
         ┌─────────────────────────────┐
         │ PPE Detector (YOLOv8-M)     │
         │ - Safety glasses/goggles    │
         │ - Hard hats                 │
         │ - Hi-vis vests              │
         │ - Work boots                │
         └─────────┬───────────────────┘
                   │
                   ↓
         ┌─────────────────────────────┐
         │ Compliance Check            │
         │ Aggregate worker PPE status │
         └─────────┬───────────────────┘
                   │
              ┌────┴────┐
              │          │
              ↓          ↓
         ┌─────────┐ ┌──────────┐
         │ Alert   │ │ Telemetry│
         │ System  │ │& Logging │
         └─────────┘ └──────────┘
```

## Module Breakdown

### 1. Inference Engines (`src/inference/`)

#### YOLODetector (`yolo_detector.py`)
- Wraps Ultralytics YOLOv8 for unified interface
- Supports PyTorch backend
- Methods:
  - `predict(image)` → Dict[boxes, confidences, class_ids]
  - `export_onnx(path)` → Exports to ONNX format

#### ONNXInference (`onnx_runtime.py`)
- Runs ONNX models via ONNX Runtime
- Supports CPU and QNN execution providers
- Methods:
  - `predict(image)` → Dict[outputs]
  - `get_input_details()`, `get_output_details()`

#### Post-processing (`postprocess.py`)
- NMS (Non-Maximum Suppression)
- Confidence filtering
- IOU-based duplicate removal

### 2. Edge Deployment (`src/edge_deployment/`)

#### Camera Handler (WIP)
- Video capture from:
  - USB cameras
  - CSI ribbon cables
  - RTSP streams
  - Video files

#### QNN Executor (WIP)
- Qualcomm QNN SDK integration
- .dlc model loading and execution
- NPU offloading on Snapdragon devices

#### Safety Monitor (WIP)
- Main inference loop
- Frame-by-frame detection
- Alert generation for non-compliant workers

#### Telemetry (WIP)
- FPS tracking
- Inference latency logging
- Results export (JSON, CSV)

### 3. Training (`src/training/`)

#### Train Worker Detector
- Fine-tune YOLOv8-N on custom warehouse data
- Transfer learning from COCO pretrained

#### Train PPE Detector
- Fine-tune YOLOv8-M for custom PPE classes
- Class imbalance handling

#### Transfer Learning Utilities
- Learning rate scheduling
- Early stopping
- Model checkpointing

### 4. Utilities (`src/utils/`)

#### Config Loader
- Load YAML configuration
- Validate settings
- Support environment variable overrides

#### Logging
- Structured logging setup
- File and console output
- Log rotation

#### Conversion & Optimization
- PyTorch → ONNX export
- ONNX → Qualcomm QNN compilation
- Model quantization (FP32 → FP16 → INT8)

## Data Flow

### Training Pipeline
```
Raw Dataset
    ↓
Preprocessing (Augmentation, Normalization)
    ↓
Train/Val/Test Split
    ↓
YOLOv8 Training
    ↓
Model Checkpoints (.pt files)
```

### Inference Pipeline
```
Input Frame
    ↓
Load Model (PyTorch/.pt or ONNX)
    ↓
Preprocessing (Resize, Normalize)
    ↓
Forward Pass
    ↓
Post-processing (NMS, Filtering)
    ↓
Output (Boxes, Classes, Confidence)
    ↓
Compliance Check
    ↓
Alert/Log/Telemetry
```

## Model Specifications

### Worker Detector
| Property | Value |
|----------|-------|
| Architecture | YOLOv8-Nano |
| Input | 640×640 RGB |
| Output | Bounding boxes + confidence |
| Classes | 1 (worker) |
| Speed | ~5-10ms per frame |

### PPE Detector
| Property | Value |
|----------|-------|
| Architecture | YOLOv8-Medium |
| Input | 640×640 RGB |
| Output | 10 class detections |
| Classes | safety glasses, goggles, hat, vest, boots, shoes |
| Speed | ~10-20ms per frame |

### Drone Classifier
| Property | Value |
|----------|-------|
| Architecture | YOLOv8-Nano (Classification) |
| Input | 224×224 RGB |
| Output | Person vs Drone probability |
| Speed | ~2-5ms per frame |

## Inference Backends

### 1. PyTorch (Development)
- **Environment**: Windows/Linux/macOS
- **Device**: CUDA (NVIDIA/Intel Arc) or CPU
- **Latency**: ~30-40ms per frame (full pipeline)
- **Memory**: ~1-2GB

### 2. ONNX Runtime (Cross-Platform)
- **Environment**: Windows/Linux/macOS
- **Device**: CPU, GPU, or QNN (with provider)
- **Latency**: ~25-35ms per frame
- **Memory**: ~0.8-1.5GB

### 3. Qualcomm QNN (Edge)
- **Environment**: Rubik Pi, Snapdragon devices
- **Device**: NPU + Hexagon accelerators
- **Latency**: ~15-25ms per frame (optimized)
- **Memory**: <500MB

## Configuration Hierarchy

```
1. Defaults (hardcoded in code)
   ↓
2. YAML Config Files (config/*.yaml)
   ↓
3. Environment Variables (overrides)
   ↓
4. Command-line Arguments (highest priority)
```

## Deployment Targets

### Development (Intel Arc 140V)
- PyTorch backend
- Full model sizes (YOLOv8-N, YOLOv8-M)
- GPU acceleration via CUDA
- ~30 FPS achievable

### Edge (Qualcomm)
- QNN backend
- Quantized models (.dlc)
- NPU acceleration
- ~30+ FPS achievable

## Performance Optimization Strategies

1. **Model Selection**
   - YOLOv8-Nano for speed-sensitive tasks
   - YOLOv8-Medium for accuracy-critical PPE
   - Skip drone classification if not needed

2. **Input Resolution**
   - Reduce from 640×640 to 480×480 for faster inference
   - Trade-off: ~15% accuracy loss

3. **Batch Processing**
   - Process 2-4 frames in parallel
   - Increase GPU utilization
   - ~1.5x throughput improvement

4. **Model Quantization**
   - Convert FP32 → FP16: ~2x speedup
   - Convert FP32 → INT8: ~4x speedup
   - Minimal accuracy loss (<1%)

5. **Caching & Tracking**
   - Skip full inference on static regions
   - Track detections across frames
   - Reduce redundant computation

## Error Handling

```
Model Load Failure
    ↓ Fall back to CPU
Input Size Mismatch
    ↓ Resize automatically
OOM (Out of Memory)
    ↓ Reduce batch size, skip inference
Inference Timeout
    ↓ Log warning, skip frame
```

## Security Considerations

- Model files stored securely (no git tracking)
- Inference logs do not contain sensitive data
- Camera streams not recorded without user consent
- All model inputs validated before processing

## Scalability

- Multi-threaded inference on multi-core systems
- Distributed inference across devices (future)
- Horizontal scaling via multiple camera feeds
- Vertical scaling via GPU/NPU utilization

## Future Enhancements

1. **Multi-Camera Fusion** — Track workers across views
2. **Temporal Tracking** — Persistent worker IDs
3. **Incident Database** — Store historical compliance
4. **Mobile Dashboard** — Real-time alerts on mobile
5. **Custom Models** — Fine-tune for specific PPE types
6. **Model Compression** — Knowledge distillation, pruning
