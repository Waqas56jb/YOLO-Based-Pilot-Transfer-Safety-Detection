from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    label: str


class YoloDetector:
    def __init__(self, model_path: str, conf: float = 0.4, iou: float = 0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)
        dets: List[Detection] = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for box, c, cls in zip(boxes, confs, clss):
                if c < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box)
                label = self.model.names[int(cls)]
                dets.append(Detection(x1, y1, x2, y2, float(c), label))
        return dets


def detect_ladder_box(frame_bgr: np.ndarray) -> Tuple[int, int, int, int] | None:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 80, minLineLength=int(0.06 * w), maxLineGap=10)
    if lines is None:
        return None
    x_bins = {}
    for l in lines[:, 0, :]:
        x1l, y1l, x2l, y2l = l
        dx, dy = abs(x2l - x1l), abs(y2l - y1l)
        if dy == 0:
            continue
        if dx / dy <= 0.2 and np.hypot(dx, dy) >= 0.12 * h:
            if x1l < int(0.4 * w) or x1l > int(0.6 * w):
                k = (x1l // 20) * 20
                x_bins.setdefault(k, []).append((x1l, y1l, x2l, y2l))
    best = max(x_bins.values(), key=len) if x_bins else None
    if not best or len(best) < 3:
        return None
    xs, ys = [], []
    for (sx1, sy1, sx2, sy2) in best:
        xs.extend([sx1, sx2])
        ys.extend([sy1, sy2])
    lx1, ly1, lx2, ly2 = max(0, min(xs) - 8), max(0, min(ys)), min(w - 1, max(xs) + 8), min(h - 1, max(ys))
    if (ly2 - ly1) > 2 * (lx2 - lx1) and (ly2 - ly1) > 0.2 * h:
        return lx1, ly1, lx2, ly2
    return None


