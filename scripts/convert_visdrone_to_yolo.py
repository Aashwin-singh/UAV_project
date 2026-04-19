from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.uav_tracking.visdrone import image_size, iter_images, parse_visdrone_line, yolo_line


def split_name(source_split: str) -> str:
    lower = source_split.lower()
    if "train" in lower:
        return "train"
    if "val" in lower:
        return "val"
    if "test" in lower:
        return "test"
    return source_split


def find_image_root(split_root: Path) -> Path:
    for candidate in (split_root / "images", split_root / "sequences", split_root):
        if candidate.exists() and any(iter_images(candidate)):
            return candidate
    raise FileNotFoundError(f"No images found under {split_root}")


def find_annotation_root(split_root: Path) -> Path:
    for candidate in (split_root / "annotations", split_root / "labels"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No annotations directory found under {split_root}")


def frame_number(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def first_annotation_width(annotation_root: Path) -> int:
    for path in annotation_root.glob("*.txt"):
        if path.stat().st_size == 0:
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            if line.strip():
                return len(line.split(","))
    return 0


def convert_per_image_annotations(image_root: Path, annotation_root: Path, out_root: Path, split: str, copy_images: bool) -> int:
    converted = 0
    images = list(iter_images(image_root))
    for image_path in tqdm(images, desc=f"Converting {split}"):
        rel = image_path.relative_to(image_root)
        annotation_path = annotation_root / rel.with_suffix(".txt")
        if not annotation_path.exists():
            annotation_path = annotation_root / f"{image_path.stem}.txt"

        out_image = out_root / "images" / split / rel
        out_label = out_root / "labels" / split / rel.with_suffix(".txt")
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)

        if copy_images:
            shutil.copy2(image_path, out_image)

        width, height = image_size(image_path)
        labels = []
        if annotation_path.exists():
            for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
                box = parse_visdrone_line(raw_line)
                if box is None:
                    continue
                label = yolo_line(box, width, height)
                if label is not None:
                    labels.append(label)
        out_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        converted += 1
    return converted


def convert_sequence_annotations(image_root: Path, annotation_root: Path, out_root: Path, split: str, copy_images: bool) -> int:
    converted = 0
    sequence_images: dict[str, dict[int, Path]] = defaultdict(dict)
    for image_path in iter_images(image_root):
        sequence = image_path.parent.name
        number = frame_number(image_path)
        if number is not None:
            sequence_images[sequence][number] = image_path

    for annotation_path in tqdm(sorted(annotation_root.glob("*.txt")), desc=f"Converting {split} sequences"):
        sequence = annotation_path.stem
        boxes_by_frame = defaultdict(list)
        for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
            box = parse_visdrone_line(raw_line)
            if box is not None and box.frame_index is not None:
                boxes_by_frame[box.frame_index].append(box)

        for frame_idx, image_path in sorted(sequence_images.get(sequence, {}).items()):
            rel = Path(sequence) / image_path.name
            out_image = out_root / "images" / split / rel
            out_label = out_root / "labels" / split / rel.with_suffix(".txt")
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)

            if copy_images:
                shutil.copy2(image_path, out_image)

            width, height = image_size(image_path)
            labels = [
                label
                for box in boxes_by_frame.get(frame_idx, [])
                if (label := yolo_line(box, width, height)) is not None
            ]
            out_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
            converted += 1
    return converted


def convert_split(split_root: Path, out_root: Path, copy_images: bool) -> int:
    split = split_name(split_root.name)
    image_root = find_image_root(split_root)
    annotation_root = find_annotation_root(split_root)

    if image_root.name == "sequences" or first_annotation_width(annotation_root) >= 9:
        return convert_sequence_annotations(image_root, annotation_root, out_root, split, copy_images)
    return convert_per_image_annotations(image_root, annotation_root, out_root, split, copy_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET/VID/MOT annotations to Ultralytics YOLO format.")
    parser.add_argument("--source", type=Path, required=True, help="Folder containing VisDrone split folders.")
    parser.add_argument("--output", type=Path, default=Path("data/visdrone_yolo"), help="Converted YOLO dataset folder.")
    parser.add_argument("--no-copy-images", action="store_true", help="Create labels only. Use if images are already copied.")
    args = parser.parse_args()

    split_roots = [
        path
        for path in sorted(args.source.iterdir())
        if path.is_dir() and any(token in path.name.lower() for token in ("train", "val", "test"))
    ]
    if not split_roots:
        split_roots = [args.source]

    total = 0
    for split_root in split_roots:
        total += convert_split(split_root, args.output, copy_images=not args.no_copy_images)

    print(f"Converted {total} images/frames into {args.output}")


if __name__ == "__main__":
    main()
