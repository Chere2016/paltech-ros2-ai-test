"""Drawing helpers for Rumex detections."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .clustering import LeafMeasurement


def _color_palette() -> list[tuple[int, int, int]]:
    # Colors chosen to be distinct on green backgrounds.
    return [
        (255, 99, 71),
        (65, 105, 225),
        (255, 215, 0),
        (72, 209, 204),
        (199, 21, 133),
        (60, 179, 113),
        (255, 140, 0),
        (106, 90, 205),
    ]


def draw_annotations(
    image: np.ndarray,
    leaves: Sequence[LeafMeasurement],
    plant_boxes: Sequence[tuple[float, float, float, float]],
    plant_centers: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Overlay leaves, masks, plant boxes, and centers on an image."""

    annotated = image.copy()
    overlay = image.copy()
    palette = _color_palette()

    for idx, leaf in enumerate(leaves):
        color = palette[idx % len(palette)]
        if leaf.polygons:
            for poly in leaf.polygons:
                pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color=color)

        x1, y1, x2, y2 = map(int, leaf.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cx, cy = map(int, leaf.center)
        cv2.circle(annotated, (cx, cy), 3, color, -1, cv2.LINE_AA)

    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    for idx, (box, center) in enumerate(zip(plant_boxes, plant_centers), start=1):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = map(int, center)
        color = (0, 255, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"P{idx}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated
