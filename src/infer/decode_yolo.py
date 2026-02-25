import torch

@torch.no_grad()
def decode_yolo(pred: torch.Tensor, conf_thres: float):
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    p = pred[0]
    xywh = p[:, 0:4]
    obj = p[:, 4:5].sigmoid()
    cls_logits = p[:, 5:]
    cls_scores, cls_ids = cls_logits.sigmoid().max(dim=1)
    scores = obj.squeeze(1) * cls_scores

    keep = scores >= conf_thres
    xywh = xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    x, y, w, h = xywh.unbind(-1)
    boxes = torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)
    return boxes, scores, cls_ids
