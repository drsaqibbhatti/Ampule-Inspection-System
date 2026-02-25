import numpy as np

def decode_to_binary_mask(pred: np.ndarray, num_classes: int = 2, threshold: float = 0.5) -> np.ndarray:
    x = pred
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        C, H, W = x.shape
        if C == 1:
            return (x[0] >= threshold).astype(np.uint8)
        cls = np.argmax(x, axis=0).astype(np.uint8)
        return (cls != 0).astype(np.uint8)
    if x.ndim == 2:
        return (x >= threshold).astype(np.uint8)
    raise ValueError(f"Unsupported seg output shape: {pred.shape}")
