"""Rumex detection pipeline built on top of a YOLO11 segmentation model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from .clustering import LeafMeasurement, cluster_leaves
from .visualization import draw_annotations


@dataclass
class PlantEstimate:
    leaf_indices: List[int]
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass
class RumexResult:
    image_path: Path
    runtime_sec: float
    leaves: List[LeafMeasurement]
    plants: List[PlantEstimate]
    output_path: Path | None = None

    @property
    def plant_centers(self) -> List[tuple[float, float]]:
        return [plant.center for plant in self.plants]

    @property
    def plant_boxes(self) -> List[tuple[float, float, float, float]]:
        return [plant.bbox for plant in self.plants]


class RumexDetector:
    """Handles inference, clustering, and visualization."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.4,
        cluster_distance_px: float | None = None,
        cluster_distance_ratio: float = 0.12,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.cluster_distance_px = cluster_distance_px
        self.cluster_distance_ratio = cluster_distance_ratio
        self.device = device

    def _cluster_threshold(self, image_shape: Sequence[int]) -> float:
        if self.cluster_distance_px is not None:
            return self.cluster_distance_px
        h, w = image_shape[:2]
        diag = float(np.hypot(w, h))
        return diag * self.cluster_distance_ratio

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> RumexResult:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        runtime, leaves, plants, annotated = self._predict(image)

        output_path = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / path.name
            cv2.imwrite(str(output_path), annotated)

        return RumexResult(
            image_path=path,
            runtime_sec=runtime,
            leaves=leaves,
            plants=plants,
            output_path=output_path,
        )

    def run_on_array(
        self,
        image: np.ndarray,
        image_name: str = "frame.png",
        output_dir: str | Path | None = None,
    ) -> RumexResult:
        runtime, leaves, plants, annotated = self._predict(image)

        output_path = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / image_name
            cv2.imwrite(str(output_path), annotated)

        return RumexResult(
            image_path=Path(image_name),
            runtime_sec=runtime,
            leaves=leaves,
            plants=plants,
            output_path=output_path,
        )

    def process_folder(
        self,
        image_dir: str | Path,
        output_dir: str | Path | None = None,
    ) -> List[RumexResult]:
        image_paths = sorted(
            [
                p
                for p in Path(image_dir).glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        return [self.process_image(path, output_dir) for path in image_paths]

    def _predict(
        self, image: np.ndarray
    ) -> tuple[float, List[LeafMeasurement], List[PlantEstimate], np.ndarray]:
        start = time.perf_counter()
        result = self.model.predict(
            image, conf=self.conf, device=self.device, verbose=False
        )[0]
        runtime = time.perf_counter() - start

        leaves = self._extract_leaves(result)
        threshold = self._cluster_threshold(image.shape)
        clusters = cluster_leaves(leaves, threshold)
        plants = self._build_plants(clusters, leaves)
        annotated = draw_annotations(
            image, leaves, [p.bbox for p in plants], [p.center for p in plants]
        )
        return runtime, leaves, plants, annotated

    def _extract_leaves(self, result) -> List[LeafMeasurement]:
        leaves: List[LeafMeasurement] = []
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return leaves

        xyxy = boxes.xyxy.cpu().numpy()
        has_masks = result.masks is not None
        mask_data = result.masks.xy if has_masks else [None] * len(xyxy)

        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            polygons = self._prepare_polygons(mask_data[idx]) if has_masks else None
            leaves.append(
                LeafMeasurement(
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    polygons=polygons,
                )
            )

        return leaves

    def _prepare_polygons(self, polygon_data) -> List[np.ndarray] | None:
        if polygon_data is None:
            return None

        if isinstance(polygon_data, np.ndarray):
            groups = [polygon_data]
        else:
            groups = polygon_data

        polygons: List[np.ndarray] = []
        for poly in groups:
            arr = np.asarray(poly, dtype=np.int32)
            if arr.size == 0:
                continue
            polygons.append(arr)

        return polygons if polygons else None

    def _build_plants(
        self, clusters: Iterable[List[int]], leaves: Sequence[LeafMeasurement]
    ) -> List[PlantEstimate]:
        plants: List[PlantEstimate] = []
        for cluster in clusters:
            xs = [leaves[idx].bbox[0] for idx in cluster] + [
                leaves[idx].bbox[2] for idx in cluster
            ]
            ys = [leaves[idx].bbox[1] for idx in cluster] + [
                leaves[idx].bbox[3] for idx in cluster
            ]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            center = (
                float(np.mean([leaves[idx].center[0] for idx in cluster])),
                float(np.mean([leaves[idx].center[1] for idx in cluster])),
            )
            plants.append(PlantEstimate(leaf_indices=cluster, bbox=bbox, center=center))
        return plants
