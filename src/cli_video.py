import argparse, json
from pathlib import Path
import cv2
from tqdm import tqdm
from src.pipeline import AmpulePipeline
from src.vis import draw_boxes, overlay_mask, draw_decision

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = AmpulePipeline.load(args.config)
    pipe = AmpulePipeline(cfg)
    names = {int(k): v for k, v in cfg["classes"]["names"].items()}

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "video_results.jsonl"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_mp4 = str(out_dir / "annotated.mp4")
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0,h0))

    with jsonl.open("w", encoding="utf-8") as f:
        it = range(nframes) if nframes>0 else iter(int,1)
        for _ in tqdm(it, total=nframes if nframes>0 else None):
            ok, frame = cap.read()
            if not ok: break
            res = pipe.run(frame)

            vis = draw_boxes(frame, res["boxes"], names)
            if res["mask_u8"] is not None:
                vis = overlay_mask(vis, res["mask_u8"], alpha=float(cfg["postprocess"]["overlay_alpha"]))
            vis = draw_decision(vis, res["decision"], res["reasons"])
            writer.write(vis)

            fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            f.write(json.dumps({"frame": fid, "decision": res["decision"], "reasons": res["reasons"], "metrics": res["metrics"]}) + "\n")

    cap.release(); writer.release()
    print("Saved:", out_mp4)

if __name__ == "__main__":
    main()
