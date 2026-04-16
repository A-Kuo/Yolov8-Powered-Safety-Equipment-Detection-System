"""Post-processing utilities for detection outputs

Functions for NMS, filtering, confidence scoring, and result formatting.
"""

from typing import Dict, List, Tuple, Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


def filter_by_confidence(
    detections: Dict[str, np.ndarray],
    confidence_threshold: float,
) -> Dict[str, np.ndarray]:
    """Filter detections by confidence threshold.

    Args:
        detections: Dictionary with 'confidences', 'boxes', 'class_ids'
        confidence_threshold: Minimum confidence score

    Returns:
        Filtered detections dictionary
    """
    confidences = detections["confidences"]
    mask = confidences >= confidence_threshold

    return {
        "boxes": detections["boxes"][mask],
        "confidences": confidences[mask],
        "class_ids": detections["class_ids"][mask],
        "class_names": detections.get("class_names", {}),
    }


def nms(
    boxes: np.ndarray,
    confidences: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Non-Maximum Suppression.

    Args:
        boxes: Nx4 array of [x1, y1, x2, y2]
        confidences: N array of confidence scores
        iou_threshold: IOU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes.T

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(confidences)[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)
