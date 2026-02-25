# Export TorchScript models to ONNX (template)
# Prereq: pip install onnx onnxruntime
# Usage:
#   python export_to_onnx.py --detector detector.ts --segmenter segmenter.ts --imgsz 640 --out_dir .
import argparse, torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", required=True)
    p.add_argument("--segmenter", default="")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()

def main():
    args = parse_args()
    x = torch.randn(1,3,args.imgsz,args.imgsz)

    det = torch.jit.load(args.detector, map_location="cpu").eval()
    torch.onnx.export(det, x, f"{args.out_dir}/detector.onnx",
                      opset_version=args.opset, input_names=["images"], output_names=["pred"],
                      dynamic_axes={"images":{0:"batch"}, "pred":{0:"batch"}})
    print("Exported detector.onnx")

    if args.segmenter:
        seg = torch.jit.load(args.segmenter, map_location="cpu").eval()
        torch.onnx.export(seg, x, f"{args.out_dir}/segmenter.onnx",
                          opset_version=args.opset, input_names=["images"], output_names=["mask"],
                          dynamic_axes={"images":{0:"batch"}, "mask":{0:"batch"}})
        print("Exported segmenter.onnx")

if __name__ == "__main__":
    main()
