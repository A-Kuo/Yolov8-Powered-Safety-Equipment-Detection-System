"""PPE Dataset Downloader

Downloads all required PPE detection datasets from Roboflow, Kaggle, and Ultralytics.
Run this script once to populate data/raw/ with all source datasets.

Usage:
    # Download all datasets
    python data/preprocessing/download_datasets.py --all

    # Download specific dataset
    python data/preprocessing/download_datasets.py --dataset roboflow_construction
    python data/preprocessing/download_datasets.py --dataset hard_hat_universe
    python data/preprocessing/download_datasets.py --dataset sh17
    python data/preprocessing/download_datasets.py --dataset ultralytics_ppe

    # Download pretrained model only
    python data/preprocessing/download_datasets.py --pretrained

Requirements:
    pip install roboflow kaggle huggingface-hub
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
MODELS_DIR = Path("models/yolo")


# ──────────────────────────────────────────────────────────────
# Ultralytics Construction-PPE (built-in, zero friction)
# ──────────────────────────────────────────────────────────────

def download_ultralytics_construction_ppe():
    """Trigger Ultralytics auto-download of construction-ppe dataset.

    This dataset is bundled with Ultralytics. Running train with the YAML
    automatically downloads it to ~/ultralytics/datasets/.
    """
    try:
        from ultralytics import YOLO
        from ultralytics.utils import DATASETS_DIR

        logger.info("Triggering Ultralytics Construction-PPE download...")
        logger.info("Dataset will be saved to: %s", DATASETS_DIR)

        # Trigger download via a dummy training call (will error on epoch 0, that's ok)
        model = YOLO("yolov8n.pt")
        try:
            model.train(data="construction-ppe.yaml", epochs=0, imgsz=640, exist_ok=True)
        except Exception:
            pass  # Training will fail at 0 epochs — dataset is already downloaded

        dest = DATASETS_DIR / "construction-ppe"
        if dest.exists():
            logger.info("✅ Construction-PPE downloaded to: %s", dest)
            return str(dest)
        else:
            # Alternative: directly download the zip
            import urllib.request
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
            zip_path = RAW_DATA_DIR / "construction-ppe.zip"
            dest_dir = RAW_DATA_DIR / "ultralytics_construction_ppe"
            dest_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading from %s ...", url)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
            zip_path.unlink()
            logger.info("✅ Construction-PPE extracted to: %s", dest_dir)
            return str(dest_dir)

    except Exception as e:
        logger.error("Failed to download Construction-PPE: %s", e)
        raise


# ──────────────────────────────────────────────────────────────
# Roboflow Datasets
# ──────────────────────────────────────────────────────────────

def _get_roboflow_api_key() -> str:
    """Get Roboflow API key from environment or prompt user."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = input(
            "\nEnter your Roboflow API key (free at https://roboflow.com): "
        ).strip()
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not set. Export it or pass via input.\n"
            "Get your free key at: https://app.roboflow.com/settings/api"
        )
    return api_key


def download_roboflow_construction_safety(api_key: str = None):
    """Download Roboflow Construction Site Safety dataset.

    ~2,801 images | Classes: Hardhat, Safety Vest, NO-Hardhat, NO-Safety Vest, Person, etc.
    Directly useful for hi-vis vest + helmet compliance detection.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Install roboflow: pip install roboflow")

    if api_key is None:
        api_key = _get_roboflow_api_key()

    dest_dir = RAW_DATA_DIR / "roboflow_construction_safety"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Roboflow Construction Site Safety dataset...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    dataset = project.version(27).download("yolov8", location=str(dest_dir))

    logger.info("✅ Construction Site Safety downloaded to: %s", dest_dir)
    return str(dest_dir)


def download_hard_hat_universe(api_key: str = None):
    """Download Hard Hat Universe dataset.

    ~7,036 images | Classes: helmet, head, person
    Largest dedicated helmet dataset — volume booster for hard_hat class.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Install roboflow: pip install roboflow")

    if api_key is None:
        api_key = _get_roboflow_api_key()

    dest_dir = RAW_DATA_DIR / "hard_hat_universe"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Hard Hat Universe dataset (~7K images)...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("universe-datasets").project("hard-hat-universe-0dy7t")
    dataset = project.version(1).download("yolov8", location=str(dest_dir))

    logger.info("✅ Hard Hat Universe downloaded to: %s", dest_dir)
    return str(dest_dir)


# ──────────────────────────────────────────────────────────────
# SH17 Dataset (Kaggle / GitHub)
# ──────────────────────────────────────────────────────────────

def download_sh17_kaggle():
    """Download SH17 dataset from Kaggle.

    8,099 images | 17 classes including gloves, shoes, safety-vest, helmet
    License: CC BY-NC-SA 4.0 (non-commercial use only)

    Requires Kaggle CLI setup:
        pip install kaggle
        # Place kaggle.json at ~/.kaggle/kaggle.json
    """
    dest_dir = RAW_DATA_DIR / "sh17"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading SH17 dataset from Kaggle...")
    logger.info("This requires Kaggle CLI configured (kaggle.json in ~/.kaggle/)")

    # Check for kaggle CLI
    import subprocess
    result = subprocess.run(["kaggle", "--version"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Configure with: https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    # Download dataset
    cmd = [
        "kaggle", "datasets", "download",
        "mugheesahmad/sh17-dataset-for-ppe-detection",
        "--path", str(dest_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Kaggle download failed: %s", result.stderr)
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")

    logger.info("✅ SH17 dataset downloaded to: %s", dest_dir)
    logger.info("⚠️  License: CC BY-NC-SA 4.0 — non-commercial use only")
    return str(dest_dir)


def download_sh17_github():
    """Download SH17 dataset from GitHub (annotations only, images from Pexels).

    NOTE: Full images require running download_from_pexels.py separately.
    """
    import subprocess

    dest_dir = RAW_DATA_DIR / "sh17"
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cloning SH17 dataset repository...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/ahmadmughees/SH17dataset", str(dest_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        raise RuntimeError(f"Git clone failed: {result.stderr}")

    logger.info("✅ SH17 annotations cloned to: %s", dest_dir)
    logger.info("ℹ  Run: cd %s && python download_from_pexels.py", dest_dir)
    return str(dest_dir)


# ──────────────────────────────────────────────────────────────
# Pretrained Models
# ──────────────────────────────────────────────────────────────

def download_keremberke_ppe_model(size: str = "m"):
    """Download pretrained YOLOv8 PPE model from Hugging Face.

    Available sizes: n (nano), s (small), m (medium)
    Classes: glove, goggles, helmet, mask, no_glove, no_goggles, no_helmet, no_mask, no_shoes, shoes

    This model is the best starting point for transfer learning on our custom classes.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Install huggingface-hub: pip install huggingface-hub")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    repo_id = f"keremberke/yolov8{size}-protective-equipment-detection"
    dest_path = MODELS_DIR / f"keremberke_ppe_{size}.pt"

    logger.info("Downloading %s from Hugging Face...", repo_id)
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="best.pt",
    )

    import shutil
    shutil.copy(model_path, dest_path)
    logger.info("✅ PPE model saved to: %s", dest_path)
    return str(dest_path)


def download_yolov8_base_models():
    """Download YOLOv8 base models (COCO pretrained) from Ultralytics.

    Downloads:
    - yolov8n.pt (Nano, 6.2MB) — worker detector, fast edge inference
    - yolov8m.pt (Medium, 49MB) — PPE detector, higher accuracy
    """
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in ["yolov8n.pt", "yolov8m.pt"]:
        dest_path = MODELS_DIR / model_name
        if dest_path.exists():
            logger.info("✅ %s already exists at %s", model_name, dest_path)
            continue

        logger.info("Downloading %s...", model_name)
        model = YOLO(model_name)  # Downloads automatically

        import shutil
        # Ultralytics downloads to working dir or ~/.cache
        if Path(model_name).exists():
            shutil.move(model_name, dest_path)
        else:
            # Model may be in ultralytics cache
            logger.info("ℹ  %s cached by Ultralytics at default location", model_name)

        logger.info("✅ %s ready", model_name)


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Download PPE detection datasets and pretrained models"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--dataset",
        choices=[
            "ultralytics_ppe",
            "roboflow_construction",
            "hard_hat_universe",
            "sh17",
        ],
        help="Specific dataset to download",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Download pretrained PPE model (keremberke + YOLOv8 base)",
    )
    parser.add_argument(
        "--roboflow-key",
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m"],
        default="m",
        help="Pretrained model size: n=nano, s=small, m=medium (default: m)",
    )
    args = parser.parse_args()

    if not any([args.all, args.dataset, args.pretrained]):
        parser.print_help()
        print(
            "\n\nQUICK START:\n"
            "  # Download the built-in Ultralytics dataset (zero friction):\n"
            "  python data/preprocessing/download_datasets.py --dataset ultralytics_ppe\n\n"
            "  # Download best pretrained model for transfer learning:\n"
            "  python data/preprocessing/download_datasets.py --pretrained\n\n"
            "  # Download everything (needs Roboflow API key + Kaggle):\n"
            "  python data/preprocessing/download_datasets.py --all\n"
        )
        return

    roboflow_key = args.roboflow_key or os.environ.get("ROBOFLOW_API_KEY")

    if args.pretrained or args.all:
        logger.info("=== Downloading pretrained models ===")
        download_yolov8_base_models()
        download_keremberke_ppe_model(size=args.model_size)

    if args.dataset == "ultralytics_ppe" or args.all:
        logger.info("=== Downloading Ultralytics Construction-PPE ===")
        download_ultralytics_construction_ppe()

    if args.dataset == "roboflow_construction" or args.all:
        logger.info("=== Downloading Roboflow Construction Safety ===")
        download_roboflow_construction_safety(api_key=roboflow_key)

    if args.dataset == "hard_hat_universe" or args.all:
        logger.info("=== Downloading Hard Hat Universe ===")
        download_hard_hat_universe(api_key=roboflow_key)

    if args.dataset == "sh17" or args.all:
        logger.info("=== Downloading SH17 Dataset ===")
        download_sh17_kaggle()

    logger.info("\n=== Download complete ===")
    logger.info("Next: python data/preprocessing/merge_datasets.py")


if __name__ == "__main__":
    main()
