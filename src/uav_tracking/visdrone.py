from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class VisDroneBox:
    frame_index: int | None
    left: float
    top: float
    width: float
    height: float
    score: float
    category: int
    truncation: int
    occlusion: int


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    return width, height


def parse_visdrone_line(line: str) -> VisDroneBox | None:
    parts = [p.strip() for p in line.split(",")]
    if not parts or all(not p for p in parts):
        return None

    values = [float(p) for p in parts if p != ""]
    if len(values) == 8:
        left, top, width, height, score, category, truncation, occlusion = values
        frame_index = None
    elif len(values) >= 10:
        frame_index = int(values[0])
        left, top, width, height = values[2:6]
        score, category, truncation, occlusion = values[6:10]
    elif len(values) == 9:
        frame_index = int(values[0])
        left, top, width, height, score, category, truncation, occlusion = values[1:9]
    else:
        raise ValueError(f"Unsupported VisDrone annotation row with {len(values)} fields: {line}")

    return VisDroneBox(
        frame_index=frame_index,
        left=left,
        top=top,
        width=width,
        height=height,
        score=score,
        category=int(category),
        truncation=int(truncation),
        occlusion=int(occlusion),
    )


def yolo_line(box: VisDroneBox, image_width: int, image_height: int) -> str | None:
    if box.category < 1 or box.category > 10:
        return None
    if box.width <= 1 or box.height <= 1:
        return None

    x_center = (box.left + box.width / 2) / image_width
    y_center = (box.top + box.height / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height

    values = [
        box.category - 1,
        min(max(x_center, 0.0), 1.0),
        min(max(y_center, 0.0), 1.0),
        min(max(width, 0.0), 1.0),
        min(max(height, 0.0), 1.0),
    ]
    return f"{values[0]} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f} {values[4]:.6f}"
