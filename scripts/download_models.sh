#!/bin/bash

# Download YOLOv8 Base Models
# This script downloads pre-trained YOLOv8 models from Ultralytics

set -e

echo "Downloading YOLOv8 base models..."

MODELS_DIR="models/yolo"
mkdir -p "$MODELS_DIR"

# Download YOLOv8-Nano (worker detection)
echo "Downloading YOLOv8-Nano..."
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt "$MODELS_DIR/"

# Download YOLOv8-Medium (PPE detection)
echo "Downloading YOLOv8-Medium..."
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
mv yolov8m.pt "$MODELS_DIR/"

echo "✓ Models downloaded successfully to $MODELS_DIR/"
echo ""
echo "Next steps:"
echo "1. Fine-tune models with your custom data:"
echo "   python src/training/train_worker_detector.py"
echo "2. Export to ONNX:"
echo "   python scripts/convert_to_onnx.py"
echo "3. Run inference:"
echo "   python scripts/run_inference_video.py --video sample.mp4"
