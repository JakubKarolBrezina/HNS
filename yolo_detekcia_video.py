#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

import cv2
import numpy as np
from ultralytics import YOLO

# Disable spam logs
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ==========================
# KONFIG
# ==========================

@dataclass
class AppConfig:
    model_path: str = "yolo11n-pose.pt"
    fov_deg: float = 60.0
    real_shoulder_m: float = 0.42
    conf_thr: float = 0.60
    safe_zone_m: float = 1.0
    warn_zone_m: float = 2.0
    draw_thick: int = 1
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    topic_state: str = "robot/safety/state"
    topic_event: str = "robot/safety/event"


class Safety(Enum):
    GO = "GO"
    SLOW = "SLOW"
    STOP = "STOP"


@dataclass
class TrackState:
    standing_height_m: Optional[float] = None
    prev_height_m: Optional[float] = None
    prev_time: float = 0.0
    fall_sus_frames: int = 0
    last_fall_time: float = 0.0


# ==========================
# POSE MATRICE
# ==========================

KP = {
    "nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4,
    "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lwri": 9, "rwri": 10,
    "lhip": 11, "rhip": 12, "lkne": 13, "rkne": 14, "lank": 15, "rank": 16
}

SKELETON = [
    (KP["lsho"], KP["lelb"]), (KP["lelb"], KP["lwri"]),
    (KP["rsho"], KP["relb"]), (KP["relb"], KP["rwri"]),
    (KP["lsho"], KP["rsho"]),
    (KP["lhip"], KP["rhip"]),
    (KP["lsho"], KP["lhip"]), (KP["rsho"], KP["rhip"]),
    (KP["lhip"], KP["lkne"]), (KP["lkne"], KP["lank"]),
    (KP["rhip"], KP["rkne"]), (KP["rkne"], KP["rank"])
]


# ==========================
# UTIL
# ==========================

def focal_len_px(img_w, fov_deg):
    return (img_w / 2) / math.tan(math.radians(fov_deg / 2))


def shoulder_px(k):
    L = k[5, 0]
    R = k[6, 0]
    if np.isnan(L) or np.isnan(R):
        return 0
    return abs(R - L)


def estimate_distance_m(shoulder_px_val, f_px, real_shoulder_m):
    if shoulder_px_val <= 10:
        return None
    return (f_px * real_shoulder_m) / shoulder_px_val


def estimate_height_m(k, s_px, real_shoulder_m):
    if s_px <= 10:
        return None
    top = k[0, 1]
    ankles = [k[15, 1], k[16, 1]]
    ankles = [x for x in ankles if not np.isnan(x)]
    if not ankles:
        return None
    px_h = max(ankles) - top
    if px_h <= 0:
        return None
    return px_h * (real_shoulder_m / s_px)


def hand_up_both(k):
    try:
        r = k[10, 1] < k[6, 1] - 40
        l = k[9, 1] < k[5, 1] - 40
        if r and l: return "both"
        if r: return "right"
        if l: return "left"
        return None
    except:
        return None


def torso_tilt_deg(k):
    try:
        sx = (k[5, 0] + k[6, 0]) / 2
        sy = (k[5, 1] + k[6, 1]) / 2
        hx = (k[11, 0] + k[12, 0]) / 2
        hy = (k[11, 1] + k[12, 1]) / 2
        return abs(math.degrees(math.atan2(sx - hx, sy - hy)))
    except:
        return None


def decide_safety(dist, cfg):
    if dist is None:
        return Safety.GO
    if dist < cfg.safe_zone_m:
        return Safety.STOP
    if dist < cfg.warn_zone_m:
        return Safety.SLOW
    return Safety.GO


# ==========================
# FALL DETECTION
# ==========================

def update_fall_state(ts, height_m, tilt, bbox, now):
    if height_m is None or tilt is None:
        ts.fall_sus_frames = 0
        ts.prev_height_m = height_m
        ts.prev_time = now
        return False

    x1, y1, x2, y2 = bbox
    horiz = (x2 - x1) / max((y2 - y1), 1)

    if tilt < 30:
        if ts.standing_height_m is None:
            ts.standing_height_m = height_m
        else:
            ts.standing_height_m = ts.standing_height_m * 0.9 + height_m * 0.1

    if ts.standing_height_m is None:
        ts.prev_height_m = height_m
        ts.prev_time = now
        return False

    too_low = height_m < 0.6 * ts.standing_height_m
    very_low = height_m < 0.45 * ts.standing_height_m

    dt = now - ts.prev_time
    big_drop = False
    if ts.prev_height_m and dt < 0.7:
        if (ts.prev_height_m - height_m) / max(ts.prev_height_m, 1e-6) > 0.35:
            big_drop = True

    suspicious = (tilt > 75 and horiz > 1.3 and (very_low or (too_low and big_drop)))

    ts.fall_sus_frames = ts.fall_sus_frames + 1 if suspicious else max(0, ts.fall_sus_frames - 1)

    if ts.fall_sus_frames >= 8 and now - ts.last_fall_time > 3:
        ts.last_fall_time = now
        ts.fall_sus_frames = 0
        return True

    ts.prev_height_m = height_m
    ts.prev_time = now
    return False


# ==========================
# DRAW FAST
# ==========================

def draw_pose_fast(f, k):
    for x, y in k:
        if not (np.isnan(x) or np.isnan(y)):
            cv2.circle(f, (int(x), int(y)), 2, (0, 0, 255), -1)

    for a, b in SKELETON:
        xa, ya = k[a]
        xb, yb = k[b]
        if not (np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb)):
            cv2.line(f, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 1)


# ==========================
# MAIN
# ==========================

def main():
    print("Vyber zdroj:")
    print("1 – Interná kamera")
    print("2 – Externá kamera")
    print("3 – Video súbor")
    ch = input("Zadaj 1/2/3: ").strip()

    if ch == "1":
        cap = cv2.VideoCapture(0)
    elif ch == "2":
        cap = cv2.VideoCapture(1)
    else:
        vid = input("Cesta k videu: ").strip()
        cap = cv2.VideoCapture(vid)

    if not cap.isOpened():
        print("KAMERA ERROR")
        return

    # --- zväčšené okno ---
    window_name = "FAST-DETECTION"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # --- rýchla kamera ---
    cap.set(3, 640)
    cap.set(4, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cfg = AppConfig()
    model = YOLO(cfg.model_path)

    track_states = {}
    prev_time = time.time()
    fps = 0

    detect_every = 2
    frame_id = 0
    last_results = None

    # FPS AVERAGE COUNTERS
    total_fps = 0.0
    frame_counter = 0

    print("SYSTEM ONLINE\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        now = time.time()
        fps = 1 / (now - prev_time + 1e-6)
        prev_time = now

        # Add to average
        total_fps += fps
        frame_counter += 1

        W = frame.shape[1]
        fpx = focal_len_px(W, cfg.fov_deg)

        # DETECT každé 2 framy
        if frame_id % detect_every == 0:
            last_results = model.predict(
                frame, conf=cfg.conf_thr, imgsz=320, verbose=False
            )

        results = last_results
        if results is None or len(results) == 0:
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        r = results[0]
        boxes = r.boxes
        kpts = r.keypoints

        for i in range(len(boxes)):
            if int(boxes.cls[i].item()) != 0:
                continue

            tid = i + 1
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            kxy = kpts.xy[i].cpu().numpy()
            kxy = np.where(np.isfinite(kxy), kxy, np.nan)

            draw_pose_fast(frame, kxy)

            s = shoulder_px(kxy)
            dist = estimate_distance_m(s, fpx, cfg.real_shoulder_m)
            height = estimate_height_m(kxy, s, cfg.real_shoulder_m)
            tilt = torso_tilt_deg(kxy)
            hand = hand_up_both(kxy)
            state = decide_safety(dist, cfg)

            ts = track_states.setdefault(tid, TrackState())
            fall = update_fall_state(ts, height, tilt, (x1, y1, x2, y2), now)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid} D:{dist}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            if hand:
                cv2.putText(frame, hand.upper() + " HAND",
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (
0, 200, 255), 2)

            if fall:
                cv2.putText(frame, "FALL!",
                            (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 3)

        # FPS DISPLAY
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord("q"):
            break

    # PRINT AVERAGE FPS
    if frame_counter > 0:
        avg_fps = total_fps / frame_counter
        print("\n============================")
        print(f"Priemerné FPS: {avg_fps:.2f}")
        print("============================\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
