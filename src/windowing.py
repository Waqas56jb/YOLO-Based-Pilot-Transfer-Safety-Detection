from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Window:
    start_idx: int
    end_idx: int
    score: float


def detect_windows(v: np.ndarray, a: np.ndarray, fps: float, Tmin_s: float = 0.4,
                   v_th: float | None = None, a_th: float | None = None) -> List[Window]:
    if len(v) == 0:
        return []
    if v_th is None:
        v_th = np.percentile(np.abs(v), 35)
    if a_th is None:
        a_th = np.percentile(np.abs(a), 35)
    Tmin = max(1, int(Tmin_s * fps))
    good = (np.abs(v) < v_th) & (np.abs(a) < a_th)
    windows: List[Window] = []
    s = None
    for i, g in enumerate(good):
        if g and s is None:
            s = i
        elif not g and s is not None:
            if i - s >= Tmin:
                score = -float(np.mean(np.abs(v[s:i])))
                windows.append(Window(s, i - 1, score))
            s = None
    if s is not None and len(good) - s >= Tmin:
        score = -float(np.mean(np.abs(v[s:len(good)])))
        windows.append(Window(s, len(good) - 1, score))
    windows.sort(key=lambda w: (w.score, -(w.end_idx - w.start_idx)))
    return windows


