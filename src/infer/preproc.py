import cv2
import numpy as np

def letterbox(img_bgr: np.ndarray, new_shape: int, color=(114,114,114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape - nw, new_shape - nh
    dw //= 2
    dh //= 2
    out = cv2.copyMakeBorder(img_resized, dh, new_shape - nh - dh, dw, new_shape - nw - dw,
                             cv2.BORDER_CONSTANT, value=color)
    return out, r, (dw, dh)

def bgr_to_chw(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = img_bgr[:, :, ::-1]
    x = img_rgb.astype(np.float32) / 255.0
    return np.transpose(x, (2,0,1))
