import torch
from torchvision.ops import nms

@torch.no_grad()
def multiclass_nms(boxes_xyxy, scores, classes, iou=0.5, conf=0.25, max_det=300):
    keep = scores >= conf
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy, scores, classes

    kept_all = []
    for c in classes.unique():
        idx = (classes == c).nonzero(as_tuple=False).squeeze(1)
        kept = nms(boxes_xyxy[idx], scores[idx], iou)
        kept_all.append(idx[kept])

    kept = torch.cat(kept_all, dim=0)
    kept = kept[scores[kept].argsort(descending=True)][:max_det]
    return boxes_xyxy[kept], scores[kept], classes[kept]
