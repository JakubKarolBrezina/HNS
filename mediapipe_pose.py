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
# LOW LIGHT ENHANCE
# ============================================================

def enhance_low_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 25)
    s = cv2.add(s, 10)
    frame = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0, (6, 6)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# ============================================================
# VIDEO SOURCE SELECT
# ============================================================

def select_video_source():
    print("\nVyber zdroj videa:")
    print("1 – Interná kamera")
    print("2 – Externá kamera")
    print("3 – Video súbor")

    ch = input("Zadaj 1 / 2 / 3: ").strip()
    if ch == "1":
        return 0, None
    elif ch == "2":
        return 1, None
    elif ch == "3":
        path = input("Cesta k videu: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError("Video neexistuje")
        return path, path
    return 0, None

# ============================================================
# MODEL DOWNLOAD
# ============================================================

MODEL_DIR = "C:/Users/Public/mp_models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")

MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float32/latest/pose_landmarker_full.task"
]

os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return MODEL_PATH

    print("Sťahujem MediaPipe model...")
    for url in MODEL_URLS:
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(MODEL_PATH) > 5_000_000:
                return MODEL_PATH
        except Exception:
            pass

    raise RuntimeError("Nepodarilo sa stiahnuť model")

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
    hand_offset: float = 0.03
    assumed_fps: int = 30

# ============================================================
# MATH
# ============================================================

def focal_len_px(W, fov):
    return (W / 2) / math.tan(math.radians(fov / 2))

def smooth(new, old, a=0.3):
    if new is None:
        return old
    if old is None:
        return new
    return old * (1 - a) + new * a

def shoulder_px(k, W):
    try:
        return abs(k[11].x - k[12].x) * W
    except:
        return None

def estimate_distance(px, f, real):
    if not px or px < 2:
        return None
    return (f * real) / px

def estimate_height(k, W, H, real):
    s = shoulder_px(k, W)
    if not s:
        return None
    mpp = real / s
    try:
        top = k[0].y * H
        ankle = max(k[27].y, k[28].y) * H
        return (ankle - top) * mpp if ankle > top else None
    except:
        return None

def hand_up(k, side, off):
    try:
        if side == "right":
            return k[16].y < (k[12].y - off)
        return k[15].y < (k[11].y - off)
    except:
        return False

# ============================================================
# FULLSCREEN FIX
# ============================================================

def fullscreen(win):
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ============================================================
# MAIN
# ============================================================

def main():
    ensure_model()
    cfg = AppConfig()

    src, file_src = select_video_source()
    cap = cv2.VideoCapture(file_src if file_src else src, cv2.CAP_DSHOW)

    cap.set(3, cfg.cam_width)
    cap.set(4, cfg.cam_height)

    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=cfg.max_persons
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    win = "MediaPipe Pose"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen(win)

    smooth_d = {}
    smooth_h = {}

    frame_idx = 0
    FRAME_TIME_MS = int(1000 / cfg.assumed_fps)

    print("READY – Q pre ukončenie")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        timestamp_ms = frame_idx * FRAME_TIME_MS  # ✅ FIX

        frame = enhance_low_light(frame)
        H, W = frame.shape[:2]
        fpx = focal_len_px(W, cfg.fov_deg)

        result = None
        if frame_idx % cfg.process_every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp_Image(mp_ImageFormat.SRGB, rgb)
            result = landmarker.detect_for_video(img, timestamp_ms)

        if result and result.pose_landmarks:
            for i, k in enumerate(result.pose_landmarks):
                smooth_d.setdefault(i, None)
                smooth_h.setdefault(i, None)

                d = estimate_distance(shoulder_px(k, W), fpx, cfg.real_shoulder_m)
                h = estimate_height(k, W, H, cfg.real_shoulder_m)

                smooth_d[i] = smooth(d, smooth_d[i])
                smooth_h[i] = smooth(h, smooth_h[i])

                x = int(k[12].x * W)
                y = int(k[12].y * H) - 10

                if smooth_d[i] is not None:
                    cv2.putText(frame, f"D: {smooth_d[i]:.2f} m",
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,255,0), 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
