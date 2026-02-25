from pathlib import Path
import yaml
import numpy as np
import cv2
import torch

from src.infer.preproc import letterbox, bgr_to_chw
from src.infer.models import OnnxModel, TorchScriptModel
from src.infer.decode_yolo import decode_yolo
from src.infer.decode_seg import decode_to_binary_mask
from src.infer.nms import multiclass_nms
from src.infer.postprocess import components_from_mask
from src.decision import decide

class AmpulePipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.names = {int(k): v for k, v in cfg["classes"]["names"].items()}

        dcfg = cfg["models"]["detector"]
        self.det_enabled = bool(dcfg.get("enabled", True))
        self.det_kind = dcfg["kind"]
        self.det_imgsz = int(dcfg["input_size"])
        self.det_conf = float(dcfg["conf"])
        self.det_iou = float(dcfg["iou"])
        self.det_max_det = int(dcfg["max_det"])

        scfg = cfg["models"]["segmenter"]
        self.seg_enabled = bool(scfg.get("enabled", False))
        self.seg_kind = scfg["kind"]
        self.seg_imgsz = int(scfg["input_size"])
        self.seg_classes = int(scfg.get("num_classes", 2))
        self.seg_th = float(scfg.get("threshold", 0.5))

        device = cfg["runtime"]["device"]
        half = bool(cfg["runtime"].get("half", False))

        self.det = None
        if self.det_enabled:
            self.det = OnnxModel(dcfg["path"], device=device) if self.det_kind == "onnx" else TorchScriptModel(dcfg["path"], device=device, half=half)

        self.seg = None
        if self.seg_enabled:
            self.seg = OnnxModel(scfg["path"], device=device) if self.seg_kind == "onnx" else TorchScriptModel(scfg["path"], device=device, half=half)

    @staticmethod
    def load(path: str):
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    def infer_detector(self, img_bgr):
        if not self.det_enabled:
            return []
        h0, w0 = img_bgr.shape[:2]
        img, r, pad = letterbox(img_bgr, self.det_imgsz)
        x = bgr_to_chw(img)
        y = self.det(x[None, ...].astype(np.float32))
        pred = torch.from_numpy(y)
        boxes, scores, cls_ids = decode_yolo(pred, conf_thres=self.det_conf)
        boxes, scores, cls_ids = multiclass_nms(boxes, scores, cls_ids, iou=self.det_iou, conf=self.det_conf, max_det=self.det_max_det)

        pad_x, pad_y = pad
        boxes = boxes.clone()
        boxes[:, [0,2]] -= pad_x
        boxes[:, [1,3]] -= pad_y
        boxes /= r
        boxes[:, 0].clamp_(0, w0); boxes[:, 2].clamp_(0, w0)
        boxes[:, 1].clamp_(0, h0); boxes[:, 3].clamp_(0, h0)

        return [{"xyxy": b.tolist(), "conf": float(s), "cls": int(c)} for b,s,c in zip(boxes.numpy(), scores.numpy(), cls_ids.numpy())]

    def infer_segmenter(self, img_bgr):
        if not self.seg_enabled:
            return None
        h0, w0 = img_bgr.shape[:2]
        img, r, pad = letterbox(img_bgr, self.seg_imgsz)
        x = bgr_to_chw(img)
        y = self.seg(x[None, ...].astype(np.float32))
        mask_lb01 = decode_to_binary_mask(y, num_classes=self.seg_classes, threshold=self.seg_th)

        pad_x, pad_y = pad
        h_crop = int(round(h0 * r))
        w_crop = int(round(w0 * r))
        mask = mask_lb01[pad_y:pad_y+h_crop, pad_x:pad_x+w_crop]
        mask = cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
        return (mask * 255).astype(np.uint8)

    def run(self, img_bgr):
        boxes = self.infer_detector(img_bgr)
        mask_u8 = self.infer_segmenter(img_bgr)
        comps = []
        if mask_u8 is not None:
            comps = components_from_mask(mask_u8, min_area=int(self.cfg["postprocess"]["min_component_area"]),
                                         max_components=int(self.cfg["postprocess"]["max_components"]))
        decision, reasons, metrics = decide(self.cfg, boxes, mask_u8, comps)
        return {"decision": decision, "reasons": reasons, "boxes": boxes, "mask_u8": mask_u8, "components": comps, "metrics": metrics}
