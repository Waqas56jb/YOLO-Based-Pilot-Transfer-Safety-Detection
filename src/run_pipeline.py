import os
import csv
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pixel import enhance_contrast_yuv, denoise_frame, unsharp_mask


@dataclass
class Window:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score: float


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def smooth(arr: np.ndarray, k: int = 9) -> np.ndarray:
    if len(arr) == 0:
        return arr
    k = max(1, k | 1)
    pad = k // 2
    padmode = "edge"
    arrp = np.pad(arr.astype(np.float32), (pad, pad), mode=padmode)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(arrp, kernel, mode="valid")


def detect_and_measure(
    video_path: str,
    model_path: str,
    out_overlay: str,
    conf: float = 0.4,
    iou: float = 0.45,
    target_size: Tuple[int, int] = (1280, 720),
) -> Tuple[List[Window], str, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = target_size

    model = YOLO(model_path)

    boat_y, ladder_y = [], []
    abs_idx = []
    frames_meta = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        # Preprocess to enhance detection in low-light/blur
        frame = denoise_frame(frame)
        frame = enhance_contrast_yuv(frame)
        frame = unsharp_mask(frame, sigma=1.0, amount=1.0, threshold=3)
        results = model(frame, conf=conf, iou=iou, verbose=False)

        boats, persons = [], []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            for box, c, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                if c < 0.3:
                    continue
                if label == "boat" or int(cls) == 8:
                    boats.append((x1, y1, x2, y2, float(c)))
                elif label == "person" or int(cls) == 0:
                    persons.append((x1, y1, x2, y2, float(c)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, 80, minLineLength=int(0.06 * width), maxLineGap=10)
        ladder_box = None
        if lines is not None:
            x_bins = {}
            for l in lines[:, 0, :]:
                x1l, y1l, x2l, y2l = l
                dx, dy = abs(x2l - x1l), abs(y2l - y1l)
                if dy == 0:
                    continue
                if dx / dy <= 0.2 and np.hypot(dx, dy) >= 0.12 * height:
                    if x1l < int(0.4 * width) or x1l > int(0.6 * width):
                        k = (x1l // 20) * 20
                        x_bins.setdefault(k, []).append((x1l, y1l, x2l, y2l))
            best = max(x_bins.values(), key=len) if x_bins else None
            if best and len(best) >= 3:
                xs, ys = [], []
                for (sx1, sy1, sx2, sy2) in best:
                    xs.extend([sx1, sx2])
                    ys.extend([sy1, sy2])
                lx1, ly1, lx2, ly2 = max(0, min(xs) - 8), max(0, min(ys)), min(width - 1, max(xs) + 8), min(height - 1, max(ys))
                if (ly2 - ly1) > 2 * (lx2 - lx1) and (ly2 - ly1) > 0.2 * height:
                    ladder_box = (lx1, ly1, lx2, ly2)

        boat_box = min(boats, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if boats else None
        boat_y_px = boat_box[1] if boat_box is not None else None
        ladder_y_px = ladder_box[3] if ladder_box is not None else None

        frames_meta.append({
            "boats": boats,
            "persons": persons,
            "ladder_box": ladder_box,
            "boat_y": boat_y_px,
            "ladder_y": ladder_y_px,
        })

        if boat_y_px is not None and ladder_y_px is not None:
            boat_y.append(float(boat_y_px))
            ladder_y.append(float(ladder_y_px))
            abs_idx.append(idx)

    cap.release()

    boat_y_s = smooth(np.array(boat_y, dtype=np.float32), k=9)
    ladder_y_s = smooth(np.array(ladder_y, dtype=np.float32), k=9)
    n = min(len(boat_y_s), len(ladder_y_s))
    if n == 0:
        return [], "", ""
    boat_y_s = boat_y_s[:n]
    ladder_y_s = ladder_y_s[:n]
    rel_y = ladder_y_s - boat_y_s
    vrel = np.diff(rel_y, prepend=rel_y[:1]) * fps
    arel = np.diff(vrel, prepend=vrel[:1]) * fps
    energy = np.sqrt(vrel ** 2)

    # Rule-based windowing
    V_th = np.percentile(np.abs(vrel), 60)
    A_th = np.percentile(np.abs(arel), 60)
    Tmin = int(0.3 * fps)
    good = (np.abs(vrel) < V_th) & (np.abs(arel) < A_th)
    windows_c = []
    s = None
    for i, g in enumerate(good):
        if g and s is None:
            s = i
        elif not g and s is not None:
            if i - s >= Tmin:
                windows_c.append((s, i - 1))
            s = None
    if s is not None and len(good) - s >= Tmin:
        windows_c.append((s, len(good) - 1))

    # Map to absolute frames and score
    windows: List[Window] = []
    for cs, ce in windows_c:
        if cs < len(abs_idx) and ce < len(abs_idx):
            s_abs, e_abs = abs_idx[cs], abs_idx[ce]
            score = float(np.mean(np.abs(vrel[cs:ce+1])))
            windows.append(Window(
                start_frame=int(s_abs),
                end_frame=int(e_abs),
                start_time=float(s_abs / fps),
                end_time=float(e_abs / fps),
                score=-score,
            ))

    # Render overlay
    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_overlay, fourcc, fps, (width, height))
    except Exception:
        writer = None

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        meta = frames_meta[idx] if 0 <= idx < len(frames_meta) else {}
        boats = meta.get("boats", [])
        persons = meta.get("persons", [])
        ladder_box = meta.get("ladder_box")

        for (x1, y1, x2, y2, c) in boats:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"boat {c:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        for (x1, y1, x2, y2, c) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"person {c:.2f}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if ladder_box is not None:
            lx1, ly1, lx2, ly2 = ladder_box
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (255, 255, 0), 2)
            cv2.putText(frame, "ladder", (lx1, max(0, ly1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # SAFE flag
        is_safe = any(w.start_frame <= idx <= w.end_frame for w in windows)
        cv2.putText(frame, "SAFE" if is_safe else "UNSAFE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0) if is_safe else (0, 0, 255), 3)

        if writer is not None:
            writer.write(frame)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Save JSON snapshot of windows and thresholds
    meta_json = os.path.splitext(out_overlay)[0] + "_windows.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump([
            {
                "start_frame": w.start_frame,
                "end_frame": w.end_frame,
                "start_time": round(w.start_time, 2),
                "end_time": round(w.end_time, 2),
                "score": w.score,
            } for w in windows
        ], f, indent=2)

    return windows, meta_json, out_overlay


def main():
    input_path = os.path.join(os.path.dirname(__file__), "..", "test.mp4")
    model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    ensure_dir(out_dir)
    out_overlay = os.path.join(out_dir, "overlay.mp4")

    windows, json_path, overlay_path = detect_and_measure(input_path, model_path, out_overlay)

    # CSV report
    csv_path = os.path.splitext(overlay_path)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_frame", "end_frame", "start_time", "end_time", "score"])
        for w in windows:
            writer.writerow([w.start_frame, w.end_frame, f"{w.start_time:.2f}", f"{w.end_time:.2f}", f"{w.score:.4f}"])

    print(f"Overlay: {overlay_path}")
    print(f"Windows JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()


