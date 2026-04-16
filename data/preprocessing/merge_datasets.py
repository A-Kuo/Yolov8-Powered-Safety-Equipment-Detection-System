"""Dataset Merger

Merges multiple source PPE datasets into a single unified dataset
using the class mappings defined in data/annotations/class_mapping.yaml.

Usage:
    python data/preprocessing/merge_datasets.py

Output:
    data/processed/merged/
        train/images/, train/labels/
        val/images/,   val/labels/
        test/images/,  test/labels/
        dataset.yaml
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/merged")
CLASS_MAPPING_FILE = Path("data/annotations/class_mapping.yaml")
TARGET_CLASSES_FILE = Path("data/annotations/classes.yaml")

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def load_class_mapping() -> Dict:
    """Load class mapping configuration."""
    with open(CLASS_MAPPING_FILE) as f:
        return yaml.safe_load(f)


def load_target_classes() -> Dict[int, str]:
    """Load target class definitions."""
    with open(TARGET_CLASSES_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]


def remap_label_file(
    label_path: Path,
    class_map: Dict[str, Optional[int]],
    source_classes: List[str],
) -> List[str]:
    """Remap class IDs in a YOLO label file to target class IDs.

    Args:
        label_path: Path to YOLO .txt label file
        class_map: Mapping from source class name to target class ID (None = discard)
        source_classes: Ordered list of source class names (index = source class ID)

    Returns:
        List of remapped label lines (empty if all annotations discarded)
    """
    if not label_path.exists():
        return []

    remapped = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            src_class_id = int(parts[0])

            if src_class_id >= len(source_classes):
                logger.warning("Class ID %d out of range in %s", src_class_id, label_path)
                continue

            src_class_name = source_classes[src_class_id]
            target_class_id = class_map.get(src_class_name)

            if target_class_id is None:
                # Discard this annotation
                continue

            # Rebuild line with remapped class ID
            remapped.append(f"{target_class_id} {' '.join(parts[1:])}")

    return remapped


def collect_samples(
    dataset_dir: Path,
    split: str = "train",
) -> List[Tuple[Path, Path]]:
    """Collect (image_path, label_path) pairs from a dataset directory.

    Handles both flat structure (all images in one dir) and
    split structure (train/val/test subdirs).
    """
    # Try split subdirectory first
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"

    if not img_dir.exists():
        # Try flat structure
        img_dir = dataset_dir / "images"
        lbl_dir = dataset_dir / "labels"

    if not img_dir.exists():
        logger.warning("No images directory found at %s", dataset_dir)
        return []

    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        samples.append((img_path, lbl_path))

    return samples


def merge_datasets(
    datasets_config: Dict,
    target_classes: Dict[int, str],
    output_dir: Path = OUTPUT_DIR,
):
    """Merge all source datasets into a unified output directory.

    Args:
        datasets_config: Loaded class_mapping.yaml content
        target_classes: Target class ID to name mapping
        output_dir: Where to write merged dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output split directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_discarded = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    random.seed(RANDOM_SEED)

    for dataset_name, dataset_cfg in datasets_config["datasets"].items():
        dataset_dir = RAW_DATA_DIR / dataset_name

        if not dataset_dir.exists():
            logger.warning("Dataset not found: %s — skipping", dataset_dir)
            continue

        class_map = dataset_cfg["class_map"]

        # Infer source classes from dataset YAML if available
        dataset_yaml = _find_dataset_yaml(dataset_dir)
        if dataset_yaml:
            source_classes = _load_source_classes(dataset_yaml)
        else:
            # Fall back to class_map keys as ordered list (fragile — warn user)
            source_classes = list(class_map.keys())
            logger.warning(
                "No dataset.yaml found for %s — using class_map order as source classes",
                dataset_name,
            )

        logger.info("Processing dataset: %s (%d source classes)", dataset_name, len(source_classes))

        # Collect all samples (try all splits)
        all_samples = []
        for split in ["train", "val", "test", ""]:
            samples = collect_samples(dataset_dir, split=split)
            all_samples.extend(samples)

        # Deduplicate
        seen = set()
        unique_samples = []
        for img_path, lbl_path in all_samples:
            if img_path not in seen:
                seen.add(img_path)
                unique_samples.append((img_path, lbl_path))

        logger.info("  Found %d unique images", len(unique_samples))

        # Shuffle and split
        random.shuffle(unique_samples)
        n = len(unique_samples)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        split_samples = {
            "train": unique_samples[:n_train],
            "val": unique_samples[n_train : n_train + n_val],
            "test": unique_samples[n_train + n_val :],
        }

        for split, samples in split_samples.items():
            for img_path, lbl_path in samples:
                # Remap labels
                remapped_lines = remap_label_file(lbl_path, class_map, source_classes)

                # Skip images with no valid annotations after remapping
                if not remapped_lines:
                    total_discarded += 1
                    continue

                # Copy image
                dest_img = output_dir / split / "images" / f"{dataset_name}_{img_path.name}"
                shutil.copy2(img_path, dest_img)

                # Write remapped labels
                dest_lbl = output_dir / split / "labels" / f"{dataset_name}_{lbl_path.stem}.txt"
                with open(dest_lbl, "w") as f:
                    f.write("\n".join(remapped_lines) + "\n")

                total_images += 1
                split_counts[split] += 1

    # Write merged dataset YAML
    _write_merged_yaml(output_dir, target_classes, split_counts)

    logger.info("\n=== Merge Complete ===")
    logger.info("Total images: %d", total_images)
    logger.info("  Train: %d", split_counts["train"])
    logger.info("  Val:   %d", split_counts["val"])
    logger.info("  Test:  %d", split_counts["test"])
    logger.info("Discarded (no valid annotations): %d", total_discarded)
    logger.info("Output: %s", output_dir)

    return output_dir


def _find_dataset_yaml(dataset_dir: Path) -> Optional[Path]:
    """Find dataset YAML file in directory."""
    for pattern in ["*.yaml", "data.yaml", "dataset.yaml"]:
        matches = list(dataset_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_source_classes(yaml_path: Path) -> List[str]:
    """Load ordered class names from a YOLO dataset YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        return names
    return []


def _write_merged_yaml(
    output_dir: Path,
    target_classes: Dict[int, str],
    split_counts: Dict[str, int],
):
    """Write dataset.yaml for the merged dataset."""
    yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(target_classes),
        "names": target_classes,
        "split_counts": split_counts,
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    logger.info("Dataset YAML written: %s", yaml_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mapping = load_class_mapping()
    target_classes = load_target_classes()

    merge_datasets(mapping, target_classes)
