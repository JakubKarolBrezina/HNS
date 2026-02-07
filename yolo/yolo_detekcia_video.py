#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from enum import Enum
from dataclasses import dataclass

# ==========================
# SILENT LOGS
# ==========================
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==========================
# CONFIG
# ==========================
@dataclass
class AppConfig:
    model_path: str = "yolo11n-pose.pt"
    conf_thr: float = 0.6
    imgsz: int = 320
    detect_every: int = 2

    fov_deg: float = 60.0
    real_shoulder_m: float = 0.42

    safe_zone_m: float = 1.0
    warn_zone_m: float = 2.0


class Safety(Enum):
    GO = "GO"
    SLOW = "SLOW"
    STOP = "STOP"


# ==========================
# KEYPOINTS
# ==========================
KP = {
    "nose": 0,
    "lsho": 5, "rsho": 6,
    "lwri": 9, "rwri": 10,
    "lhip": 11, "rhip": 12,
    "lank": 15, "rank": 16
}

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# ==========================
# UTILS
# ==========================
def focal_len_px(w, fov):
    return (w / 2) / math.tan(math.radians(fov / 2))


def smooth(prev, new, alpha=0.2):
    if new is None:
        return prev
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


def smooth_kpts(prev, new, alpha=0.15):
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


def shoulder_px(k):
    l, r = k[KP["lsho"], 0], k[KP["rsho"], 0]
    if np.isnan(l) or np.isnan(r):
        return 0
    return abs(r - l)


def estimate_distance(s_px, fpx, cfg):
    if s_px < 15:
        return None
    return (fpx * cfg.real_shoulder_m) / s_px


def decide_safety(dist, cfg):
    if dist is None:
        return Safety.GO
    if dist < cfg.safe_zone_m:
        return Safety.STOP
    if dist < cfg.warn_zone_m:
        return Safety.SLOW
    return Safety.GO


def hand_up(k, body_h):
    try:
        sy = (k[KP["lsho"], 1] + k[KP["rsho"], 1]) / 2
        thresh = 0.15 * body_h

        left = k[KP["lwri"], 1] < sy - thresh
        right = k[KP["rwri"], 1] < sy - thresh

        if left and right:
            return "BOTH-HANDS"
        if left:
            return "LEFT-HAND"
        if right:
            return "RIGHT-HAND"
        return None
    except Exception:
        return None


def torso_tilt_deg(k):
    try:
        sx = (k[5, 0] + k[6, 0]) / 2
        sy = (k[5, 1] + k[6, 1]) / 2
        hx = (k[11, 0] + k[12, 0]) / 2
        hy = (k[11, 1] + k[12, 1]) / 2
        return abs(math.degrees(math.atan2(sx - hx, sy - hy)))
    except Exception:
        return None


def draw_pose(frame, k):
    for x, y in k:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    for a, b in SKELETON:
        xa, ya = k[a]
        xb, yb = k[b]
        if not any(map(np.isnan, [xa, ya, xb, yb])):
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     (255, 255, 255), 2)


# ==========================
# MAIN
# ==========================
def main():
    print("Vyber zdroj:")
    print("1 – Interná kamera")
    print("2 – Externá kamera")
    print("3 – Video súbor")

    ch = input("Zadaj 1/2/3: ").strip()
    cap = cv2.VideoCapture(0 if ch == "1" else 1 if ch == "2" else input("Cesta k videu: ").strip())

    if not cap.isOpened():
        print("KAMERA ERROR")
        return

    cap.set(3, 640)
    cap.set(4, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window = "PRO VISION"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    cfg = AppConfig()
    model = YOLO(cfg.model_path)

    frame_id = 0
    last_results = None
    prev_kpts = None
    standing_height_px = None
    fall_counter = 0

    prev_time = time.time()
    fps = 0

    print("\nSYSTEM ONLINE\n")

    fpx = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % 10 == 0:
            now = time.time()
            fps = 10 / (now - prev_time + 1e-6)
            prev_time = now

        if fpx is None:
            fpx = focal_len_px(frame.shape[1], cfg.fov_deg)

        if frame_id % cfg.detect_every == 0:
            last_results = model.predict(
                frame,
                imgsz=cfg.imgsz,
                conf=cfg.conf_thr,
                verbose=False,
                device=0,
                half=True
            )

        if last_results and len(last_results) > 0 and last_results[0].boxes is not None:
            r = last_results[0]
            if len(r.boxes) > 0:
                areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * \
                        (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
                idx = areas.argmax().item()

                box = r.boxes.xyxy[idx].cpu().numpy().astype(int)
                k_raw = r.keypoints.xy[idx].cpu().numpy()
                k_raw = np.where(np.isfinite(k_raw), k_raw, np.nan)

                prev_kpts = smooth_kpts(prev_kpts, k_raw)
                k = prev_kpts

                draw_pose(frame, k)

                nose_y = k[KP["nose"], 1]
                ankles = [k[KP["lank"], 1], k[KP["rank"], 1]]
                ankles = [a for a in ankles if not np.isnan(a)]

                fall = False
                body_h = None

                if ankles and not np.isnan(nose_y):
                    body_h = max(ankles) - nose_y
                    standing_height_px = smooth(standing_height_px, body_h, 0.05)
                    ratio = body_h / max(standing_height_px, 1)
                    tilt = torso_tilt_deg(k)

                    bw = box[2] - box[0]
                    bh = box[3] - box[1]

                    fall = tilt and tilt > 75 and ratio < 0.55 and bw / max(bh, 1) > 1.3

                fall_counter = fall_counter + 1 if fall else max(0, fall_counter - 1)
                fall = fall_counter >= 3

                dist = estimate_distance(shoulder_px(k), fpx, cfg)
                safety = decide_safety(dist, cfg)
                hand = hand_up(k, body_h if body_h else 200)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                if hand:
                    cv2.putText(frame, hand, (box[0], box[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                if fall:
                    cv2.putText(frame, "FALL!", (box[0], box[1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                cv2.putText(frame, safety.value, (box[0], box[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow(window, frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
