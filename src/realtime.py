import argparse, time
import cv2
from src.pipeline import AmpulePipeline
from src.vis import draw_boxes, overlay_mask, draw_decision

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default="0")
    ap.add_argument("--record", default="")
    args = ap.parse_args()

    cfg = AmpulePipeline.load(args.config)
    pipe = AmpulePipeline(cfg)
    names = {int(k): v for k, v in cfg["classes"]["names"].items()}

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open source")

    writer = None
    if args.record:
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w0,h0))

    prev = time.time(); fps_ema = 0.0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = pipe.run(frame)

        vis = draw_boxes(frame, res["boxes"], names)
        if res["mask_u8"] is not None:
            vis = overlay_mask(vis, res["mask_u8"], alpha=float(cfg["postprocess"]["overlay_alpha"]))
        vis = draw_decision(vis, res["decision"], res["reasons"])

        now = time.time(); dt = now - prev; prev = now
        fps = 1.0/max(dt,1e-6)
        fps_ema = fps if fps_ema==0 else 0.9*fps_ema+0.1*fps
        cv2.putText(vis, f"FPS: {fps_ema:.1f}", (20, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if writer is not None: writer.write(vis)
        cv2.imshow("Ampule AOI", vis)
        if (cv2.waitKey(1) & 0xFF) == 27: break

    cap.release()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
