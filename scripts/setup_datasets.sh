#!/bin/bash

# PPE Dataset Setup Script
# Downloads and prepares all datasets for training.
#
# Usage:
#   bash scripts/setup_datasets.sh
#
# Requirements:
#   - Python virtual environment active (source edge-env/bin/activate)
#   - ROBOFLOW_API_KEY environment variable set (free at https://roboflow.com)
#   - Kaggle CLI configured (~/.kaggle/kaggle.json) for SH17

set -e

echo "============================================================"
echo "  YOLOv8 PPE Safety Detection — Dataset Setup"
echo "============================================================"
echo ""

# Check environment
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  No virtual environment detected."
    echo "   Run: source edge-env/bin/activate"
    echo ""
fi

# Install data acquisition dependencies
echo "[1/5] Installing data dependencies..."
pip install -q roboflow kaggle huggingface-hub albumentations

echo ""
echo "[2/5] Downloading Ultralytics Construction-PPE (zero setup, built-in)..."
python data/preprocessing/download_datasets.py --dataset ultralytics_ppe

echo ""
echo "[3/5] Downloading pretrained PPE model (keremberke YOLOv8m)..."
python data/preprocessing/download_datasets.py --pretrained --model-size m

echo ""
echo "[4/5] Downloading Roboflow datasets (requires ROBOFLOW_API_KEY)..."
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "   ROBOFLOW_API_KEY not set."
    echo "   Get your free key at: https://app.roboflow.com/settings/api"
    echo "   Then run:"
    echo "     export ROBOFLOW_API_KEY=your_key_here"
    echo "     python data/preprocessing/download_datasets.py --dataset roboflow_construction"
    echo "     python data/preprocessing/download_datasets.py --dataset hard_hat_universe"
    echo "   Skipping Roboflow downloads..."
else
    python data/preprocessing/download_datasets.py --dataset roboflow_construction
    python data/preprocessing/download_datasets.py --dataset hard_hat_universe
fi

echo ""
echo "[5/5] Downloading SH17 dataset (requires Kaggle CLI)..."
if command -v kaggle &> /dev/null; then
    python data/preprocessing/download_datasets.py --dataset sh17
else
    echo "   Kaggle CLI not found."
    echo "   Install: pip install kaggle"
    echo "   Configure: https://github.com/Kaggle/kaggle-api#api-credentials"
    echo "   Then run: python data/preprocessing/download_datasets.py --dataset sh17"
    echo "   Skipping SH17 download..."
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Merge datasets:"
echo "     python data/preprocessing/merge_datasets.py"
echo ""
echo "  2. Start training:"
echo "     python src/training/train_ppe_detector.py"
echo ""
echo "  3. Or use pretrained model directly:"
echo "     python scripts/run_inference_video.py --video sample.mp4"
echo "============================================================"
