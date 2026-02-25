import cv2
import numpy as np

def components_from_mask(mask_u8: np.ndarray, min_area: int = 30, max_components: int = 200):
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i].tolist()
        if area < min_area:
            continue
        comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area_px": int(area)})
    comps.sort(key=lambda d: d["area_px"], reverse=True)
    return comps[:max_components]
