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
# LOW LIGHT
# ============================================================

def enhance_low_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[...,2] = cv2.add(hsv[...,2], 20)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ============================================================
# VIDEO SOURCE
# ============================================================

def select_video_source():
    print("1 – Interná kamera")
    print("2 – Externá kamera")
    print("3 – Video súbor")
    ch = input("Vyber: ").strip()
    if ch == "1": return 0, None
    if ch == "2": return 1, None
    if ch == "3":
        p = input("Cesta k videu: ").strip()
        return p, p
    return 0, None

# ============================================================
# MODEL DOWNLOAD
# ============================================================

MODEL_DIR = "C:/Users/Public/mp_models"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return
    print("Sťahujem model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH,"wb") as f:
        for c in r.iter_content(8192):
            f.write(c)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class AppConfig:
    cam_width: int = 1280
    cam_height: int = 720
    fov_deg: float = 60
    real_shoulder_m: float = 0.42
    max_persons: int = 5
    process_every_n: int = 2
    assumed_fps: int = 30

# ============================================================
# SKELETON
# ============================================================

SKELETON = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28)
]

def draw_skeleton(frame,k,W,H):
    for a,b in SKELETON:
        pa,pb=k[a],k[b]
        cv2.line(frame,(int(pa.x*W),int(pa.y*H)),
                        (int(pb.x*W),int(pb.y*H)),(0,255,255),2)
    for p in k:
        cv2.circle(frame,(int(p.x*W),int(p.y*H)),3,(0,255,0),-1)

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

    landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=cfg.max_persons
        )
    )

    frame_idx = 0
    FRAME_TIME_MS = int(1000 / cfg.assumed_fps)
    last_result = None

    print("RUNNING – Q")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        timestamp_ms = frame_idx * FRAME_TIME_MS

        if frame_idx % 2 == 0:
            frame = enhance_low_light(frame)

        H,W = frame.shape[:2]

        if frame_idx % cfg.process_every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp_Image(mp_ImageFormat.SRGB, rgb)
            res = landmarker.detect_for_video(img, timestamp_ms)
            if res.pose_landmarks:
                last_result = res.pose_landmarks

        if last_result:
            for k in last_result:
                draw_skeleton(frame,k,W,H)

        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(1)&0xFF in (ord('q'),ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
