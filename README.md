# Automated Ampule Inspection System (Python)

Deep learning-based ampule inspection system for defect detection and quality control (detection + optional segmentation), built with Python, PyTorch, and OpenCV.

Project reference: https://drsaqibbhatti.com/projects/ampule-inspection.html

## Features
- Image/folder inference + JSONL logging
- Video inference (annotated MP4 + JSONL)
- Real-time camera/RTSP
- Optional FastAPI `/inspect` endpoint
- PASS/FAIL decision rules (crack, foreign material, bubble size, fill-level)

## Install
```bash
cd python
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
Images:
```bash
python -m src.cli_images --config config.yaml --source path/to/images --out outputs --save_overlay
```

Video:
```bash
python -m src.cli_video --config config.yaml --video path/to/video.mp4 --out outputs
```

Realtime:
```bash
python -m src.realtime --config config.yaml --source 0
```

API (optional):
```bash
python -m src.server --config config.yaml
```

## Plug in models
Put your exported models in `python/models/` and update `python/config.yaml`.
If output shapes differ, edit:
- `src/infer/decode_yolo.py`
- `src/infer/decode_seg.py`
