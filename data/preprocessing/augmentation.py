"""Data Augmentation Pipeline

Augmentation strategy tuned for PPE detection:
- Hi-vis vest: brightness/contrast (important — vests in shadows may look like regular clothing)
- Hard hat: rotation, perspective (detected from various camera angles)
- Small objects (goggles, boots): scale-up crops to boost small object representation
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    import cv2
    import numpy as np
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    logger.warning("albumentations not installed. Install with: pip install albumentations")


def get_ppe_augmentation_pipeline(
    image_size: int = 640,
    is_training: bool = True,
) -> "A.Compose":
    """Build augmentation pipeline tuned for PPE detection.

    Key considerations:
    - Hi-vis vests: color jitter + brightness to simulate shadows
    - Hard hats: rotation to detect from different camera angles
    - Small objects (goggles, boots): scale transforms to improve small object detection
    - No aggressive color transforms that remove hi-vis vest's yellow/orange color

    Args:
        image_size: Target image size
        is_training: If True, apply all augmentations. If False, only resize/normalize.

    Returns:
        Albumentations Compose pipeline with bbox support
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("Install albumentations: pip install albumentations")

    if not is_training:
        return A.Compose(
            [A.Resize(image_size, image_size)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            # Flip — horizontal only (vertical flips make hard hats look weird)
            A.HorizontalFlip(p=0.5),

            # Brightness/Contrast — Critical for hi-vis vest in different lighting
            # Hi-vis vest must remain detectable even in shadows or glare
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7,
            ),

            # HSV color jitter — mild to preserve hi-vis yellow/orange color
            # Too aggressive would make hi-vis vest look like regular clothing
            A.HueSaturationValue(
                hue_shift_limit=10,   # Minimal hue shift
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.5,
            ),

            # Scale + rotation — detect workers at different distances and angles
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.3,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),

            # Perspective transform — different camera angles (top-down, eye-level)
            A.Perspective(scale=(0.05, 0.10), p=0.3),

            # Blur — simulate motion or defocus
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.GaussianBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),

            # Noise — simulate sensor noise on edge devices
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # Shadow — simulate partial shadowing (common in warehouses)
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=4,
                p=0.3,
            ),

            # JPEG compression artifacts — simulate real camera quality
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),

            # Cutout — random rectangles to simulate occlusion
            # Important for partial PPE detection (e.g. partially visible vest)
            A.CoarseDropout(
                max_holes=4,
                max_height=32,
                max_width=32,
                fill_value=0,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,  # Require 30% of bbox visible after augmentation
        ),
    )


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augmentation_factor: int = 3,
    image_size: int = 640,
):
    """Augment a YOLO-format dataset by a given factor.

    Args:
        input_dir: Input dataset directory (with train/images, train/labels)
        output_dir: Output directory for augmented dataset
        augmentation_factor: How many augmented copies per original image
        image_size: Target image size
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("Install albumentations: pip install albumentations")

    import cv2
    import numpy as np

    pipeline = get_ppe_augmentation_pipeline(image_size=image_size, is_training=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)

    img_dir = input_dir / "images"
    lbl_dir = input_dir / "labels"

    image_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    total = len(image_paths) * augmentation_factor

    logger.info(
        "Augmenting %d images × %d = %d total",
        len(image_paths), augmentation_factor, total,
    )

    processed = 0
    for img_path in image_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read image: %s", img_path)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = _load_yolo_labels(lbl_path)

        for aug_idx in range(augmentation_factor):
            try:
                transformed = pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                aug_image = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_labels = transformed["class_labels"]

                if not aug_bboxes:
                    continue  # Skip if all bboxes were removed

                # Save augmented image
                out_name = f"{img_path.stem}_aug{aug_idx}"
                out_img = output_dir / "images" / f"{out_name}.jpg"
                cv2.imwrite(
                    str(out_img),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

                # Save augmented labels
                out_lbl = output_dir / "labels" / f"{out_name}.txt"
                _save_yolo_labels(out_lbl, aug_bboxes, aug_labels)

                processed += 1

            except Exception as e:
                logger.warning("Augmentation failed for %s: %s", img_path, e)

    logger.info("✅ Generated %d augmented images", processed)


def _load_yolo_labels(label_path: Path) -> Tuple[List, List]:
    """Load YOLO format labels from file."""
    bboxes = []
    class_labels = []

    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)

    return bboxes, class_labels


def _save_yolo_labels(label_path: Path, bboxes: List, class_labels: List):
    """Save YOLO format labels to file."""
    with open(label_path, "w") as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
