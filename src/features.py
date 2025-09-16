import numpy as np


def smooth_mean(arr: np.ndarray, k: int = 9) -> np.ndarray:
    if len(arr) == 0:
        return arr
    k = max(1, k | 1)
    pad = k // 2
    arrp = np.pad(arr.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arrp, kernel, mode="valid")


def derivatives(y: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    v = np.diff(y, prepend=y[:1]) * fps
    a = np.diff(v, prepend=v[:1]) * fps
    return v, a


def motion_energy(v: np.ndarray, window: int) -> np.ndarray:
    if len(v) == 0:
        return v
    window = max(1, window)
    pad = window - 1
    arrp = np.pad(v.astype(np.float32) ** 2, (pad, 0), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    rms2 = np.convolve(arrp, kernel, mode="valid")
    return np.sqrt(rms2)


