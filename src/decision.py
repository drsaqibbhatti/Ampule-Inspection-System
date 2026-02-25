def _name_to_ids(names: dict):
    out = {}
    for k, v in names.items():
        try:
            out[str(v)] = int(k)
        except Exception:
            pass
    return out

def compute_metrics_from_boxes(boxes, class_ids):
    metrics = {"bubble_diam_px_max": 0.0, "has_crack": False, "has_fill_low": False, "foreign_count": 0}
    bubble_id = class_ids.get("bubble", -1)
    crack_id = class_ids.get("crack", -1)
    fill_low_id = class_ids.get("fill_level_low", -1)
    foreign_ids = {class_ids.get("particle",-999), class_ids.get("contamination",-999)}

    for b in boxes:
        cls = int(b["cls"])
        x1,y1,x2,y2 = b["xyxy"]
        w = max(0.0, x2-x1)
        h = max(0.0, y2-y1)
        if cls == bubble_id:
            metrics["bubble_diam_px_max"] = max(metrics["bubble_diam_px_max"], float(max(w,h)))
        if cls == crack_id:
            metrics["has_crack"] = True
        if cls == fill_low_id:
            metrics["has_fill_low"] = True
        if cls in foreign_ids:
            metrics["foreign_count"] += 1
    return metrics

def decide(cfg: dict, boxes: list, mask_u8, comps: list):
    names = cfg["classes"]["names"]
    class_ids = _name_to_ids(names)
    metrics = compute_metrics_from_boxes(boxes, class_ids)

    reasons = []
    if metrics["has_crack"]:
        reasons.append("crack_detected")

    foreign_total_area = int(sum(c["area_px"] for c in comps)) if comps else 0
    if metrics["foreign_count"] > 0:
        reasons.append("foreign_material_detected")
    if foreign_total_area > int(cfg["postprocess"]["min_component_area"]) * 5:
        reasons.append("foreign_material_area_high")

    if metrics["bubble_diam_px_max"] > float(cfg["rules"]["bubble_max_diameter_px"]):
        reasons.append("bubble_too_large")

    if metrics["has_fill_low"]:
        reasons.append("fill_level_low")

    decision = "PASS" if not reasons else "FAIL"
    metrics["foreign_total_area_px"] = foreign_total_area
    return decision, reasons, metrics
