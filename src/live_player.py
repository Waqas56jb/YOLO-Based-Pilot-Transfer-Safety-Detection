import os
import cv2
import numpy as np
from typing import Tuple

from .detector import YoloDetector, detect_ladder_box


def play_live(video_path: str, model_path: str, conf: float = 0.4, iou: float = 0.45,
              target_size: Tuple[int, int] = (1280, 720)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = target_size
    detector = YoloDetector(model_path, conf=conf, iou=iou)

    cv2.namedWindow("DSS Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DSS Live", width, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

        dets = detector.detect(frame)
        boats = [(d.x1, d.y1, d.x2, d.y2, d.conf) for d in dets if d.label == "boat"]
        persons = [(d.x1, d.y1, d.x2, d.y2, d.conf) for d in dets if d.label == "person"]
        ladder_box = detect_ladder_box(frame)

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

        cv2.imshow("DSS Live", frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    vid = os.path.join(here, "..", "test.mp4")
    mdl = os.path.join(here, "..", "yolov8n.pt")
    play_live(vid, mdl)


