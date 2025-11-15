#!/usr/bin/env python3
"""CLI script for running Rumex plant detection on a folder of images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from paltech_rumex import RumexDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rumex detection pipeline.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the YOLO11 segmentation weights.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory that contains test images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Where to store annotated outputs.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for YOLO inference.",
    )
    parser.add_argument(
        "--cluster-distance",
        type=float,
        default=None,
        help="Distance threshold in pixels for clustering leaves (optional).",
    )
    parser.add_argument(
        "--cluster-ratio",
        type=float,
        default=0.12,
        help="Relative diagonal ratio for clustering if distance is not provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. 'cpu', '0').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = RumexDetector(
        model_path=args.model,
        conf=args.conf,
        cluster_distance_px=args.cluster_distance,
        cluster_distance_ratio=args.cluster_ratio,
        device=args.device,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    results = detector.process_folder(args.images, args.output)
    if not results:
        print("No images were processed.")
        return

    for res in results:
        print(
            f"{res.image_path.name}: "
            f"{len(res.leaves)} leaves -> {len(res.plants)} plants, "
            f"{res.runtime_sec:.3f}s"
        )
        for idx, center in enumerate(res.plant_centers, start=1):
            cx, cy = center
            print(f"  Plant {idx} center: ({cx:.1f}, {cy:.1f})")
        if res.output_path:
            print(f"  Saved: {res.output_path}")


if __name__ == "__main__":
    main()
