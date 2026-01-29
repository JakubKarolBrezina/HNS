#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import time
import math
import numpy as np
import requests
import platform
from dataclasses import dataclass

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as mp_Image, ImageFormat as mp_ImageFormat

# ============================================================
# LOW-LIGHT (optimalizované – nie každý frame)
# ============================================================

def enhance_low_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 20)
    s = cv2.add(s, 8)
    hsv = cv2.merge([h, s, v])
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0, (6, 6)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class AppConfig:
    cam_width: int = 1280
    cam_height: int = 720
    fov_deg: float = 60.0
    real_shoulder_m: float = 0.42
    max_persons: int = 5
    process_every_n: int = 2
    low_light_every_n: int = 2
    hand_offset: float = 0.03

# ============================================================
# MATH
# ============================================================

def focal_len_px(w, fov):
    return (w / 2) / math.tan(math.radians(fov / 2))

def smooth(new, old, a=0.3):
    if new is None:
        return old
    if old is None:
        return new
    return old * (1 - a) + new * a

def shoulder_px(k, W):
    try:
        return abs(k[11].x - k[12].x) * W
    except Exception:
        return None

def estimate_distance(px, f, real):
    if not px or px < 2:
        return None
    return max(0.2, (f * real) / px)

def estimate_height(k, W, H, real):
    s = shoulder_px(k, W)
    if not s:
        return None
    mpp = real / s
    try:
        top = k[0].y * H
        ankle = max(k[27].y, k[28].y) * H
        return (ankle - top) * mpp if ankle > top else None
    except Exception:
        return None

def hand_up(k, side, off):
    try:
        if side == "right":
            return k[16].y < (k[12].y - off)
        return k[15].y < (k[11].y - off)
    except Exception:
        return False

# ============================================================
# MAIN
# ============================================================

def main():
    cfg = AppConfig()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, cfg.cam_width)
    cap.set(4, cfg.cam_height)

    base = mp_python.BaseOptions(model_asset_path="C:/Users/Public/mp_models/pose_landmarker_full.task")
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=cfg.max_persons
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    smoothers = {}
    frame_id = 0
    last_ts = time.time()
    fps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1

        if frame_id % cfg.low_light_every_n == 0:
            frame = enhance_low_light(frame)

        H, W = frame.shape[:2]
        fpx = focal_len_px(W, cfg.fov_deg)

        if frame_id % cfg.process_every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp_Image(mp_ImageFormat.SRGB, rgb)
            ts = int((time.time() - last_ts) * 1000)
            last_ts = time.time()
            result = landmarker.detect_for_video(img, ts)
        else:
            result = None

        if result and result.pose_landmarks:
            for i, k in enumerate(result.pose_landmarks):
                if i not in smoothers:
                    smoothers[i] = {"d": None, "h": None}

                d = estimate_distance(shoulder_px(k, W), fpx, cfg.real_shoulder_m)
                h = estimate_height(k, W, H, cfg.real_shoulder_m)

                smoothers[i]["d"] = smooth(d, smoothers[i]["d"])
                smoothers[i]["h"] = smooth(h, smoothers[i]["h"])

                rx = int(k[12].x * W)
                ry = int(k[12].y * H) - 10

                if smoothers[i]["d"]:
                    cv2.putText(frame, f"D {smoothers[i]['d']:.2f}m",
                                (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        now = time.time()
        fps = 1 / (now - last_ts + 1e-6)
        last_ts = now

        cv2.putText(frame, f"FPS {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,200,0), 2)

        cv2.imshow("MediaPipe Pose OPT", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



