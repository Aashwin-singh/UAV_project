# UAV Object Detection and Motion Tracking

This project trains a YOLO detector on VisDrone UAV imagery and runs multi-object tracking with ByteTrack.

## 1. Project Structure

```text
UAV_Project/
  configs/
    visdrone.yaml
    tracker_bytetrack.yaml
  data/
    raw/                 # put downloaded VisDrone folders here
    visdrone_yolo/       # generated YOLO-format dataset
  scripts/
    check_setup.py
    convert_visdrone_to_yolo.py
    train_yolo.py
    validate_yolo.py
    track_bytetrack.py
  src/uav_tracking/
    visdrone.py
  outputs/
  runs/
```

## 2. Create Conda Environment

For an NVIDIA GPU:

```powershell
conda env create -f environment.yml
conda activate uav-yolo-bytetrack
python scripts/check_setup.py
```

If your CUDA version does not match `pytorch-cuda=12.1`, install PyTorch from the official selector, then run:

```powershell
pip install -r requirements.txt
```

For CPU-only work:

```powershell
conda create -n uav-yolo-bytetrack python=3.11 -y
conda activate uav-yolo-bytetrack
pip install -r requirements.txt
```

## 3. Download VisDrone

Recommended datasets:

- Object detection in videos: `VisDrone2019-VID-train`, `VisDrone2019-VID-val`
- Multi-object tracking: `VisDrone2019-MOT-train`, `VisDrone2019-MOT-val`
- Static image detection, if you also want more detector data: `VisDrone2019-DET-train`, `VisDrone2019-DET-val`

Place them like this:

```text
data/raw/
  VisDrone2019-VID-train/
  VisDrone2019-VID-val/
```

The converter also supports the common DET image layout.

## 4. Convert VisDrone to YOLO Format

```powershell
python scripts/convert_visdrone_to_yolo.py --source data/raw --output data/visdrone_yolo
```

Output layout:

```text
data/visdrone_yolo/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

YOLO label format:

```text
class_id x_center y_center width height
```

Coordinates are normalized to `0..1`. VisDrone classes `1..10` are mapped to YOLO classes `0..9`; ignored regions and `others` are skipped.

## 5. Train YOLO

Start small first:

```powershell
python scripts/train_yolo.py --model yolo11n.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

If your Ultralytics install supports YOLO26 weights and you want to try the newest family:

```powershell
python scripts/train_yolo.py --model yolo26n.pt --epochs 50 --imgsz 640 --batch 8 --device 0
```

Expected trained weights:

```text
runs/detect/visdrone_yolo/weights/best.pt
```

## 6. Validate Detection

```powershell
python scripts/validate_yolo.py --weights runs/detect/visdrone_yolo/weights/best.pt --device 0
```

Watch:

- `mAP50`
- `mAP50-95`
- per-class AP
- confusion matrix in `runs/detect/`

## 7. Run ByteTrack Motion Tracking

```powershell
python scripts/track_bytetrack.py `
  --weights runs/detect/visdrone_yolo/weights/best.pt `
  --source data/raw/your_test_video.mp4 `
  --output outputs/tracked_video.mp4 `
  --tracker configs/tracker_bytetrack.yaml `
  --conf 0.20 `
  --iou 0.50 `
  --device 0
```

ByteTrack links detections across frames and assigns stable track IDs. The script also draws short motion trails.

## 8. Practical Training Tips

- Use `yolo11n.pt` or `yolo26n.pt` first to confirm the pipeline.
- Move to `s` or `m` model sizes after the converter, labels, and validation are correct.
- UAV objects are small, so try `--imgsz 960` or `--imgsz 1280` if GPU memory allows.
- If tracking drops IDs during occlusion, increase `track_buffer` in `configs/tracker_bytetrack.yaml`.
- If false tracks appear, raise `track_high_thresh` or `new_track_thresh`.
- If small objects are missed, lower `--conf` during tracking, but expect more false positives.

## 9. Milestones

1. Environment check passes.
2. Dataset converts and label counts look reasonable.
3. Train nano model for 5 epochs as a smoke test.
4. Train full run for 50-100 epochs.
5. Validate and inspect confusion matrix.
6. Run ByteTrack on VisDrone videos.
7. Tune detector confidence and tracker thresholds.
8. Export model for deployment if needed.
