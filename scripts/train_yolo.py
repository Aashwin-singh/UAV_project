from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on converted VisDrone data.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model, e.g. yolo11n.pt, yolo26n.pt, or a local .pt file.")
    parser.add_argument("--data", default="configs/visdrone.yaml", help="Ultralytics dataset YAML.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0, help="GPU index like 0, or cpu.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="visdrone_yolo")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=25,
        pretrained=True,
        cache=False,
        workers=4,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training complete. Best weights expected at: {best}")


if __name__ == "__main__":
    main()
