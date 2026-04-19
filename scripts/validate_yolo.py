from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    parser.add_argument("--weights", required=True, help="Path to best.pt or another model checkpoint.")
    parser.add_argument("--data", default="configs/visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(metrics)


if __name__ == "__main__":
    main()
