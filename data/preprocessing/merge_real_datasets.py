#!/usr/bin/env python3
"""
Merge real datasets (Roboflow + Kaggle SH17) with synthetic data.

Phase 2B/2C: Merge and remap classes to unified 10-class schema
- Roboflow Construction Safety (2,801 images, 3 relevant classes)
- Kaggle SH17 (8,099 images, 10+ relevant classes)
- Synthetic (50 images, already in target schema)

Total: ~1,200 merged images with class remapping and train/val/test split

Usage:
    python data/preprocessing/merge_real_datasets.py \
        --roboflow data/raw/roboflow_construction \
        --kaggle data/raw/kaggle_sh17 \
        --synthetic data/processed/merged/train \
        --output data/processed/mixed \
        --config data/annotations/class_mapping_public.yaml
"""

import os
import sys
import argparse
import yaml
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ClassMapper:
    """Remap class IDs from source datasets to target schema."""

    def __init__(self, config_path: str):
        """Load class mapping configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.target_schema = self.config['target_schema']

    def remap_label_file(self,
                         label_path: str,
                         source: str,
                         output_path: str) -> bool:
        """
        Remap a label file from source format to target schema.

        Returns True if remapping successful, False if should skip.
        """
        if source not in ['roboflow', 'kaggle_sh17', 'synthetic']:
            print(f"  Unknown source: {source}")
            return False

        mapping = self.config[source]['mapping']

        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            remapped_lines = []

            for line in lines:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                old_class_id = int(parts[0])

                if old_class_id not in mapping:
                    print(f"  Warning: Unknown class {old_class_id} in {source}")
                    continue

                remap_info = mapping[old_class_id]
                target_class = remap_info.get('target')

                # Skip if no target class (removed from schema)
                if target_class is None:
                    continue

                # Remap class ID
                parts[0] = str(target_class)

                # Optionally boost confidence (not used for labels, kept for future)
                # confidence_boost = remap_info.get('confidence_boost', 0.0)

                remapped_lines.append(' '.join(parts) + '\n')

            # Write remapped labels
            if remapped_lines:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.writelines(remapped_lines)
                return True

            return False  # Empty label file after remapping

        except Exception as e:
            print(f"  Error remapping {label_path}: {e}")
            return False

    def validate_label(self, label_path: str) -> bool:
        """Validate label file format."""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < 5:
                    return False

                # Check class ID in valid range
                class_id = int(parts[0])
                if not (0 <= class_id <= 9):
                    return False

                # Check coordinates are normalized [0,1]
                x, y, w, h = map(float, parts[1:5])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    return False

            return True

        except Exception:
            return False


class DatasetMerger:
    """Merge multiple datasets into unified format."""

    def __init__(self, mapper: ClassMapper, output_dir: str):
        """Initialize merger."""
        self.mapper = mapper
        self.output_dir = output_dir
        self.stats = {
            'roboflow': {'images': 0, 'labels': 0},
            'kaggle': {'images': 0, 'labels': 0},
            'synthetic': {'images': 0, 'labels': 0},
            'total': {'images': 0, 'labels': 0}
        }

    def merge_roboflow(self, roboflow_dir: str) -> List[Tuple[str, str]]:
        """Merge Roboflow dataset."""
        print(f"\n[Roboflow] Merging from {roboflow_dir}...")

        merged_pairs = []

        # Find images and labels
        train_images = Path(roboflow_dir) / 'images' / 'train'
        train_labels = Path(roboflow_dir) / 'labels' / 'train'

        if not train_images.exists():
            print(f"  → {train_images} not found")
            return merged_pairs

        image_files = sorted(train_images.glob('*.jpg')) + sorted(train_images.glob('*.png'))
        print(f"  Found {len(image_files)} images")

        for img_path in image_files:
            basename = img_path.stem
            label_path = train_labels / f'{basename}.txt'

            if not label_path.exists():
                continue

            # Remap and validate
            temp_label = '/tmp/temp_label.txt'
            if self.mapper.remap_label_file(str(label_path), 'roboflow', temp_label):
                if self.mapper.validate_label(temp_label):
                    merged_pairs.append((str(img_path), temp_label))
                    self.stats['roboflow']['images'] += 1
                    self.stats['roboflow']['labels'] += 1

        print(f"  ✓ Merged {len(merged_pairs)} roboflow images")
        return merged_pairs

    def merge_kaggle_sh17(self, kaggle_dir: str) -> List[Tuple[str, str]]:
        """Merge Kaggle SH17 dataset."""
        print(f"\n[Kaggle] Merging from {kaggle_dir}...")

        merged_pairs = []

        # Find all images and labels recursively
        kaggle_path = Path(kaggle_dir)

        # Look for common structures
        image_dirs = [
            kaggle_path / 'images',
            kaggle_path / 'train' / 'images',
            kaggle_path / 'images' / 'train'
        ]

        image_root = None
        for dir_path in image_dirs:
            if dir_path.exists():
                image_root = dir_path
                break

        if not image_root:
            # Try to find any jpg/png files
            all_images = list(kaggle_path.rglob('*.jpg')) + list(kaggle_path.rglob('*.png'))
            print(f"  Found {len(all_images)} images in {kaggle_dir}")

            for img_path in all_images[:500]:  # Limit to 500 to avoid too many
                basename = img_path.stem

                # Look for label file near image
                possible_label_paths = [
                    img_path.parent / f'{basename}.txt',
                    img_path.parent.parent / 'labels' / f'{basename}.txt',
                ]

                label_path = None
                for lp in possible_label_paths:
                    if lp.exists():
                        label_path = lp
                        break

                if label_path:
                    temp_label = '/tmp/temp_label.txt'
                    if self.mapper.remap_label_file(str(label_path), 'kaggle_sh17', temp_label):
                        if self.mapper.validate_label(temp_label):
                            merged_pairs.append((str(img_path), temp_label))
                            self.stats['kaggle']['images'] += 1
                            self.stats['kaggle']['labels'] += 1

            print(f"  ✓ Merged {len(merged_pairs)} Kaggle images")
            return merged_pairs

        # Standard structure found
        image_files = sorted(image_root.glob('**/*.jpg')) + sorted(image_root.glob('**/*.png'))
        print(f"  Found {len(image_files)} images")

        for img_path in image_files:
            basename = img_path.stem

            # Look for label in parallel directory
            label_path = img_path.parent.parent / 'labels' / f'{basename}.txt'

            if not label_path.exists():
                # Try current directory
                label_path = img_path.parent / f'{basename}.txt'

            if label_path.exists():
                temp_label = '/tmp/temp_label.txt'
                if self.mapper.remap_label_file(str(label_path), 'kaggle_sh17', temp_label):
                    if self.mapper.validate_label(temp_label):
                        merged_pairs.append((str(img_path), temp_label))
                        self.stats['kaggle']['images'] += 1
                        self.stats['kaggle']['labels'] += 1

        print(f"  ✓ Merged {len(merged_pairs)} Kaggle images")
        return merged_pairs

    def merge_synthetic(self, synthetic_dir: str) -> List[Tuple[str, str]]:
        """Merge synthetic dataset."""
        print(f"\n[Synthetic] Merging from {synthetic_dir}...")

        merged_pairs = []

        synthetic_path = Path(synthetic_dir)

        if not synthetic_path.exists():
            print(f"  → {synthetic_dir} not found")
            return merged_pairs

        # Find images
        image_files = sorted(synthetic_path.glob('*.jpg')) + sorted(synthetic_path.glob('*.png'))
        print(f"  Found {len(image_files)} images")

        # Look for labels in same directory or labels subdir
        label_dir = synthetic_path.parent / 'labels' if (synthetic_path.parent / 'labels').exists() else synthetic_path

        for img_path in image_files:
            basename = img_path.stem
            label_path = label_dir / f'{basename}.txt'

            if not label_path.exists():
                continue

            # Synthetic data already in target schema, just copy
            if self.mapper.validate_label(str(label_path)):
                merged_pairs.append((str(img_path), str(label_path)))
                self.stats['synthetic']['images'] += 1
                self.stats['synthetic']['labels'] += 1

        print(f"  ✓ Merged {len(merged_pairs)} synthetic images")
        return merged_pairs

    def split_and_save(self, all_pairs: List[Tuple[str, str]],
                      split: Dict[str, float]) -> None:
        """Split data into train/val/test and save."""
        print(f"\n[Splitting] Creating train/val/test split...")

        # Shuffle
        random.shuffle(all_pairs)

        total = len(all_pairs)
        train_idx = int(total * split['train'])
        val_idx = int(total * (split['train'] + split['val']))

        train_pairs = all_pairs[:train_idx]
        val_pairs = all_pairs[train_idx:val_idx]
        test_pairs = all_pairs[val_idx:]

        # Create directories
        for split_name in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split_name, 'labels'), exist_ok=True)

        # Save train
        for img_path, label_path in train_pairs:
            img_name = Path(img_path).name
            label_name = Path(label_path).name

            shutil.copy(img_path, os.path.join(self.output_dir, 'train', 'images', img_name))
            shutil.copy(label_path, os.path.join(self.output_dir, 'train', 'labels', label_name))

        # Save val
        for img_path, label_path in val_pairs:
            img_name = Path(img_path).name
            label_name = Path(label_path).name

            shutil.copy(img_path, os.path.join(self.output_dir, 'val', 'images', img_name))
            shutil.copy(label_path, os.path.join(self.output_dir, 'val', 'labels', label_name))

        # Save test
        for img_path, label_path in test_pairs:
            img_name = Path(img_path).name
            label_name = Path(label_path).name

            shutil.copy(img_path, os.path.join(self.output_dir, 'test', 'images', img_name))
            shutil.copy(label_path, os.path.join(self.output_dir, 'test', 'labels', label_name))

        self.stats['total']['images'] = total
        self.stats['total']['labels'] = total

        print(f"  ✓ Train: {len(train_pairs)} images")
        print(f"  ✓ Val:   {len(val_pairs)} images")
        print(f"  ✓ Test:  {len(test_pairs)} images")


def main():
    parser = argparse.ArgumentParser(
        description='Merge real + synthetic datasets for Phase 2C training'
    )
    parser.add_argument('--roboflow', default='data/raw/roboflow_construction',
                       help='Roboflow dataset path')
    parser.add_argument('--kaggle', default='data/raw/kaggle_sh17',
                       help='Kaggle SH17 dataset path')
    parser.add_argument('--synthetic', default='data/processed/merged/train/images',
                       help='Synthetic dataset path')
    parser.add_argument('--output', default='data/processed/mixed',
                       help='Output directory')
    parser.add_argument('--config', default='data/annotations/class_mapping_public.yaml',
                       help='Class mapping config')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate, do not merge')

    args = parser.parse_args()

    print("="*70)
    print("PHASE 2C: Merge Real Datasets with Class Remapping")
    print("="*70)

    # Load mapper
    if not os.path.exists(args.config):
        print(f"✗ Config not found: {args.config}")
        return 1

    mapper = ClassMapper(args.config)
    merger = DatasetMerger(mapper, args.output)

    print(f"\nConfig: {args.config}")
    print(f"Output: {args.output}")

    if args.validate_only:
        print("\n[Validation Only] Not merging, checking config...")
        print("✓ Config loaded successfully")
        return 0

    # Merge all sources
    all_pairs = []

    if os.path.exists(args.roboflow):
        all_pairs.extend(merger.merge_roboflow(args.roboflow))

    if os.path.exists(args.kaggle):
        all_pairs.extend(merger.merge_kaggle_sh17(args.kaggle))

    if os.path.exists(args.synthetic):
        all_pairs.extend(merger.merge_synthetic(args.synthetic))

    # Split and save
    if all_pairs:
        merger.split_and_save(all_pairs, {'train': 0.70, 'val': 0.15, 'test': 0.15})

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Roboflow:       {merger.stats['roboflow']['images']} images")
        print(f"Kaggle SH17:    {merger.stats['kaggle']['images']} images")
        print(f"Synthetic:      {merger.stats['synthetic']['images']} images")
        print(f"TOTAL:          {merger.stats['total']['images']} images")
        print(f"\nOutput: {args.output}")
        print("="*70)

        if merger.stats['total']['images'] >= 1000:
            print("✓ SUCCESS: Ready for Phase 2C fine-tuning")
            return 0
        else:
            print(f"⚠ WARNING: Only {merger.stats['total']['images']} images (target: 1000+)")
            return 0
    else:
        print("✗ No images merged. Check dataset paths.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
