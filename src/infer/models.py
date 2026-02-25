import numpy as np
import torch
import onnxruntime as ort

class OnnxModel:
    def __init__(self, path: str, device: str = "cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray):
        return self.sess.run(None, {self.in_name: x})[0]

class TorchScriptModel:
    def __init__(self, path: str, device: str = "cuda", half: bool = False):
        self.device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.model = torch.jit.load(path, map_location="cpu").eval()
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.half = bool(half) and self.device == "cuda"

    def __call__(self, x: np.ndarray):
        xt = torch.from_numpy(x)
        if self.device == "cuda":
            xt = xt.cuda()
        if self.half:
            xt = xt.half()
        with torch.no_grad():
            y = self.model(xt)
        if isinstance(y, (list, tuple)):
            y = y[0]
        return y.detach().cpu().numpy()
