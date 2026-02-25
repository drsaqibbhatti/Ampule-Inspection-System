import argparse, json
from pathlib import Path
import cv2
from tqdm import tqdm
from src.pipeline import AmpulePipeline
from src.vis import draw_boxes, overlay_mask, draw_decision

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def iter_images(src: Path):
    if src.is_file() and src.suffix.lower() in IMG_EXTS:
        yield src
    elif src.is_dir():
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        raise ValueError(f"Unsupported source: {src}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--save_overlay", action="store_true")
    args = ap.parse_args()

    cfg = AmpulePipeline.load(args.config)
    pipe = AmpulePipeline(cfg)
    names = {int(k): v for k, v in cfg["classes"]["names"].items()}

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "results.jsonl"

    with jsonl.open("w", encoding="utf-8") as f:
        paths = list(iter_images(Path(args.source)))
        for p in tqdm(paths):
            img = cv2.imread(str(p))
            if img is None: 
                continue
            res = pipe.run(img)

            vis = draw_boxes(img, res["boxes"], names)
            if res["mask_u8"] is not None:
                vis = overlay_mask(vis, res["mask_u8"], alpha=float(cfg["postprocess"]["overlay_alpha"]))
            vis = draw_decision(vis, res["decision"], res["reasons"])

            if args.save_overlay:
                cv2.imwrite(str(out_dir / f"{p.stem}_overlay.jpg"), vis)
            if res["mask_u8"] is not None:
                cv2.imwrite(str(out_dir / f"{p.stem}_mask.png"), res["mask_u8"])

            f.write(json.dumps({"file": str(p), "decision": res["decision"], "reasons": res["reasons"], "metrics": res["metrics"]}) + "\n")

    print("Saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
