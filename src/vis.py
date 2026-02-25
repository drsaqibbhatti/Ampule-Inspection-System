import cv2
import numpy as np

def draw_boxes(img_bgr, boxes, names):
    out = img_bgr.copy()
    for b in boxes:
        x1,y1,x2,y2 = map(int, b["xyxy"])
        cls = int(b["cls"]); conf = float(b["conf"])
        label = str(names.get(cls, cls))
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out

def overlay_mask(img_bgr, mask_u8, alpha=0.45):
    if mask_u8 is None:
        return img_bgr
    out = img_bgr.copy()
    m = (mask_u8 > 0).astype(np.uint8) * 255
    colored = np.zeros_like(out)
    colored[:, :, 1] = m
    return cv2.addWeighted(out, 1.0, colored, alpha, 0)

def draw_decision(img_bgr, decision, reasons):
    out = img_bgr.copy()
    color = (0,255,0) if decision == "PASS" else (0,0,255)
    cv2.putText(out, decision, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    y = 75
    for r in reasons[:6]:
        cv2.putText(out, f"- {r}", (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28
    return out
