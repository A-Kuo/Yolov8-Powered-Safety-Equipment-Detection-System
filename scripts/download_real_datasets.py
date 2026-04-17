#!/usr/bin/env python3
"""
Download real PPE safety datasets from public sources.

Phase 2B: Real Data Collection
- Roboflow Construction Safety (2,801 images, free, no auth)
- Kaggle SH17 Dataset (8,099 images, requires kaggle.json)

Usage:
    python scripts/download_real_datasets.py [--roboflow] [--kaggle] [--dry-run]

Examples:
    python scripts/download_real_datasets.py                # Download both
    python scripts/download_real_datasets.py --roboflow     # Roboflow only
    python scripts/download_real_datasets.py --kaggle       # Kaggle only
    python scripts/download_real_datasets.py --dry-run      # Show what would download
"""

import os
import sys
import argparse
import shutil
from pathlib import Path


def download_roboflow(output_dir: str, dry_run: bool = False) -> bool:
    """Download Roboflow Construction Safety dataset (2,801 images)."""
    print("\n[Roboflow] Downloading construction-site-safety dataset...")
    print("  Source: Roboflow Universe (free, no authentication)")
    print("  Images: ~2,801 construction safety images")
    print("  Classes: Hardhat, Safety Vest, Person")

    if dry_run:
        print("  [DRY RUN] Would download to:", output_dir)
        return True

    try:
        from roboflow import Roboflow

        output_path = os.path.join(output_dir, 'roboflow_construction')
        os.makedirs(output_path, exist_ok=True)

        print(f"  Downloading to: {output_path}")

        # Free tier API key
        rf = Roboflow(api_key="roboflow_free")
        project = rf.workspace("roboflow-universe").project("construction-site-safety-yolov8")
        dataset = project.download("yolov8", location=output_path)

        print(f"  ✓ PASS: Roboflow dataset downloaded successfully")

        # Count images
        images_path = os.path.join(output_path, 'images', 'train')
        if os.path.exists(images_path):
            img_count = len([f for f in os.listdir(images_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  ✓ Total images: {img_count}")

        return True

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        print("  → Try installing: pip install roboflow")
        return False


def download_kaggle(output_dir: str, dry_run: bool = False) -> bool:
    """Download Kaggle SH17 dataset (8,099 images)."""
    print("\n[Kaggle] Downloading SH17 PPE safety dataset...")
    print("  Source: Kaggle Dataset (free, requires kaggle.json)")
    print("  Images: ~8,099 diverse PPE images")
    print("  Classes: 17 safety equipment classes")

    if dry_run:
        print("  [DRY RUN] Would download to:", output_dir)
        return True

    try:
        import kaggle

        output_path = os.path.join(output_dir, 'kaggle_sh17')
        os.makedirs(output_path, exist_ok=True)

        print(f"  Downloading to: {output_path}")

        # Download from Kaggle API
        kaggle.api.dataset_download_files(
            'deeptech/sh17-dataset',
            path=output_path,
            unzip=True
        )

        print(f"  ✓ PASS: Kaggle SH17 dataset downloaded successfully")

        # Count images
        images_count = 0
        for root, dirs, files in os.walk(output_path):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images_count += 1

        if images_count > 0:
            print(f"  ✓ Total images: {images_count}")

        return True

    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        print("\n  Setup Kaggle authentication:")
        print("  1. Go to: https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New API Token' (saves kaggle.json)")
        print("  3. Place ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("  5. Try again: python scripts/download_real_datasets.py --kaggle")
        return False


def validate_downloads(output_dir: str) -> dict:
    """Validate downloaded datasets."""
    print("\n[Validation] Checking downloaded datasets...")

    results = {
        'roboflow': {'success': False, 'count': 0},
        'kaggle': {'success': False, 'count': 0},
        'total': 0
    }

    # Check Roboflow
    roboflow_path = os.path.join(output_dir, 'roboflow_construction')
    if os.path.exists(roboflow_path):
        images_path = os.path.join(roboflow_path, 'images')
        if os.path.exists(images_path):
            count = 0
            for root, dirs, files in os.walk(images_path):
                count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if count > 0:
                results['roboflow'] = {'success': True, 'count': count}
                print(f"  ✓ Roboflow: {count} images")

    # Check Kaggle
    kaggle_path = os.path.join(output_dir, 'kaggle_sh17')
    if os.path.exists(kaggle_path):
        count = 0
        for root, dirs, files in os.walk(kaggle_path):
            count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if count > 0:
            results['kaggle'] = {'success': True, 'count': count}
            print(f"  ✓ Kaggle SH17: {count} images")

    results['total'] = results['roboflow']['count'] + results['kaggle']['count']

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Download real PPE safety datasets for Phase 2'
    )
    parser.add_argument('--roboflow', action='store_true',
                       help='Download Roboflow dataset only')
    parser.add_argument('--kaggle', action='store_true',
                       help='Download Kaggle dataset only')
    parser.add_argument('--output', default='data/raw',
                       help='Output directory (default: data/raw)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be downloaded without downloading')

    args = parser.parse_args()

    # Determine what to download
    download_both = not (args.roboflow or args.kaggle)

    print("="*70)
    print("PHASE 2B: Real Data Collection - Download Datasets")
    print("="*70)

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    success_count = 0

    # Download Roboflow
    if download_both or args.roboflow:
        if download_roboflow(output_dir, args.dry_run):
            success_count += 1

    # Download Kaggle
    if download_both or args.kaggle:
        if download_kaggle(output_dir, args.dry_run):
            success_count += 1

    # Validate
    if not args.dry_run:
        results = validate_downloads(output_dir)

        print("\n" + "="*70)
        print(f"SUMMARY")
        print("="*70)
        print(f"Roboflow:      {results['roboflow']['count']:,} images {'✓' if results['roboflow']['success'] else '✗'}")
        print(f"Kaggle SH17:   {results['kaggle']['count']:,} images {'✓' if results['kaggle']['success'] else '✗'}")
        print(f"TOTAL:         {results['total']:,} images")

        if results['total'] >= 1000:
            print("\n✓ SUCCESS: Sufficient real data for Phase 2C fine-tuning (>1000 images)")
        elif results['total'] > 0:
            print(f"\n⚠ WARNING: {results['total']:,} images collected, target is 1000+")
        else:
            print("\n✗ FAIL: No images downloaded. Check network and authentication.")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
