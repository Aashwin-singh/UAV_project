from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection with ByteTrack motion tracking on a video.")
    parser.add_argument("--weights", required=True, help="Trained YOLO .pt model, e.g. runs/detect/.../best.pt")
    parser.add_argument("--source", required=True, help="Video path, webcam index, image folder, or stream URL.")
    parser.add_argument("--output", default="outputs/tracked_video.mp4")
    parser.add_argument("--tracker", default="configs/tracker_bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default=0)
    parser.add_argument("--trail", type=int, default=30, help="Number of recent center points to draw for each track.")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    track_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    writer = None

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for result in results:
        frame = result.plot()

        if result.boxes is not None and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x, y, _, _ = box
                history = track_history[track_id]
                history.append((float(x), float(y)))
                if len(history) > args.trail:
                    history.pop(0)
                points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

        writer.write(frame)

    if writer is not None:
        writer.release()
    print(f"Saved tracked output to {output_path}")


if __name__ == "__main__":
    main()
