import argparse, base64
from pathlib import Path
import yaml
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from src.pipeline import AmpulePipeline
from src.vis import draw_boxes, overlay_mask, draw_decision

app = FastAPI(title="Ampule Inspection API")
PIPE = None
CFG = None

class Box(BaseModel):
    cls: int
    label: str
    conf: float
    xyxy: list[float]

class InspectResponse(BaseModel):
    decision: str
    reasons: list[str]
    metrics: dict
    boxes: list[Box]
    mask_png_base64: str | None = None
    overlay_jpg_base64: str | None = None

def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def bytes_to_bgr(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/inspect", response_model=InspectResponse)
async def inspect(file: UploadFile = File(...), return_mask: bool = False, return_overlay: bool = False):
    img = bytes_to_bgr(await file.read())
    res = PIPE.run(img)
    names = {int(k): v for k, v in CFG["classes"]["names"].items()}

    resp = {
        "decision": res["decision"],
        "reasons": res["reasons"],
        "metrics": res["metrics"],
        "boxes": [
            {"cls": int(b["cls"]), "label": names.get(int(b["cls"]), str(b["cls"])),
             "conf": float(b["conf"]), "xyxy": [float(x) for x in b["xyxy"]]}
            for b in res["boxes"]
        ],
        "mask_png_base64": None,
        "overlay_jpg_base64": None,
    }

    if return_mask and res["mask_u8"] is not None:
        ok, buf = cv2.imencode(".png", res["mask_u8"])
        if ok:
            resp["mask_png_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")

    if return_overlay:
        vis = draw_boxes(img, res["boxes"], names)
        if res["mask_u8"] is not None:
            vis = overlay_mask(vis, res["mask_u8"], alpha=float(CFG["postprocess"]["overlay_alpha"]))
        vis = draw_decision(vis, res["decision"], res["reasons"])
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            resp["overlay_jpg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")

    return resp

def main():
    global PIPE, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    CFG = load_cfg(args.config)
    PIPE = AmpulePipeline(CFG)

    import uvicorn
    uvicorn.run("src.server:app", host=str(CFG["server"]["host"]), port=int(CFG["server"]["port"]), reload=False)

if __name__ == "__main__":
    main()
