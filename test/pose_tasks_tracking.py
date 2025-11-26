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
from typing import Optional, List, Tuple, Dict

# MediaPipe Tasks (vision)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as mp_Image, ImageFormat as mp_ImageFormat

# ============================================================
# 🔽 MODEL DOWNLOAD (SAFE)
# ============================================================

MODEL_DIR = "C:/Users/Public/mp_models"
MODEL_PATH = os.path.join(MODEL_DIR, "../pose_landmarker_full.task")

MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float32/latest/pose_landmarker_full.task",
]

EXPECTED_MIN = 5_000_000
EXPECTED_MAX = 50_000_000

os.makedirs(MODEL_DIR, exist_ok=True)

def ensure_model(path: str = MODEL_PATH):
    if os.path.exists(path):
        size = os.path.getsize(path)
        if EXPECTED_MIN < size < EXPECTED_MAX:
            print("✅ Model exists and is valid.")
            return path
        else:
            print("⚠️ Model corrupted. Re-downloading...")
            try: os.remove(path)
            except: pass

    print("⬇️ Downloading MediaPipe model...")
    last_err = None

    for url in MODEL_URLS:
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)

            size = os.path.getsize(path)
            if EXPECTED_MIN < size < EXPECTED_MAX:
                print("✅ Model downloaded OK.")
                return path

            print(f"⚠️ Wrong file size: {size} — trying next URL.")
        except Exception as e:
            last_err = e
            print(f"⚠️ Failed: {e}")

    raise RuntimeError(f"❌ Cannot download model. Last error: {last_err}")

# ============================================================
# ⚙️ CONFIG
# ============================================================

@dataclass
class AppConfig:
    cam_index: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    fov_deg: float = 60.0
    real_shoulder_m: float = 0.42
    max_persons: int = 5
    draw_thick: int = 2
    hand_offset: float = 0.03
    process_every_n: int = 2   # ➜ MediaPipe beží každé 2 snímky

# ============================================================
# KĹBY / SKELETON
# ============================================================

MP_JOINTS = ["nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer","left_ear","right_ear",
    "mouth_left","mouth_right","left_shoulder","right_shoulder","left_elbow",
    "right_elbow","left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb","left_hip",
    "right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"]

SKELETON = [
    (11,12), (11,13),(13,15), (12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27), (24,26),(26,28)
]

LABEL_JOINTS = [0, 11, 12, 15, 16, 23, 24, 27, 28]

# ============================================================
# MATEMATIKA
# ============================================================

def focal_len_px(W, fov): return (W/2)/math.tan(math.radians(fov/2))

def shoulder_px(k, W):
    try: return abs(k[11].x - k[12].x) * W
    except: return 0

def estimate_distance(px, f, real):
    if px <= 2: return None
    return (f * real) / px

def estimate_height(k, W, H, real):
    s = shoulder_px(k, W)
    if s <= 2: return None
    mpp = real / s
    try:
        top = k[0].y * H
        ankle = max(k[27].y*H, k[28].y*H)
        ph = ankle - top
        if ph <= 0: return None
        return ph * mpp
    except:
        return None

def hand_up(k, side, off):
    try:
        if side == "right":
            return k[16].y < k[12].y - off
        return k[15].y < k[11].y - off
    except:
        return False

# ============================================================
# DRAW
# ============================================================

def draw_skeleton(frame, k, W, H, thick=2, draw_labels=True):
    for i,p in enumerate(k):
        x,y = int(p.x*W), int(p.y*H)
        cv2.circle(frame,(x,y),4,(0,255,0),-1)
        if draw_labels and i in LABEL_JOINTS:
            cv2.putText(frame, MP_JOINTS[i], (x+6, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    for a,b in SKELETON:
        pa,pb=k[a],k[b]
        cv2.line(frame,
            (int(pa.x*W), int(pa.y*H)),
            (int(pb.x*W), int(pb.y*H)),
            (0,200,255), thick)

def put_info(frame, pos, dist, height, r_up, l_up):
    x0,y0 = pos
    dy=24
    i=0
    if dist is not None:
        cv2.putText(frame,f"Dist: {dist:.2f} m",(x0,y0+i*dy),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,255,180),2); i+=1
    if height is not None:
        cv2.putText(frame,f"Height: {height:.2f} m",(x0,y0+i*dy),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2); i+=1
    if r_up:
        cv2.putText(frame,"Right hand: UP",(x0,y0+i*dy),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2); i+=1
    if l_up:
        cv2.putText(frame,"Left hand: UP",(x0,y0+i*dy),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2); i+=1

# ============================================================
# FULLSCREEN FIX
# ============================================================

def fullscreen(win):
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if platform.system()=="Windows":
        try:
            import ctypes
            hwnd=ctypes.windll.user32.FindWindowW(None,win)
            if hwnd:
                ctypes.windll.user32.SetWindowLongW(hwnd,-16,0x80000000)
                ctypes.windll.user32.ShowWindow(hwnd,3)
        except:
            pass

# ============================================================
# MAIN
# ============================================================

def main():
    ensure_model()
    cfg = AppConfig()

    cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    win = "Optimized MediaPipe Pose"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    f_px = None
    last_W = None
    frame_idx = 0
    timestamp_ms = 0
    last_result = None

    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=cfg.max_persons,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    prev_t = time.time()
    fullscreen(win)

    print("🚀 READY – press Q to exit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        if W != last_W:
            f_px = focal_len_px(W, cfg.fov_deg)
            last_W = W

        frame_idx += 1

        # ------- MediaPipe every N frames -------
        if frame_idx % cfg.process_every_n == 0 or last_result is None:
            img = mp_Image(image_format=mp_ImageFormat.SRGB, data=frame)
            result = landmarker.detect_for_video(img, timestamp_ms)
            last_result = result
            timestamp_ms += int(1000/30)
        else:
            result = last_result

        persons = result.pose_landmarks or []

        cv2.putText(frame, f"People: {len(persons)}", (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)

        for idx, kpts in enumerate(persons):

            draw_skeleton(frame, kpts, W, H,
                thick=cfg.draw_thick,
                draw_labels=(idx==0))   # text iba pre prvého človeka

            s = shoulder_px(kpts, W)
            dist = estimate_distance(s, f_px, cfg.real_shoulder_m)
            height = estimate_height(kpts, W, H, cfg.real_shoulder_m)
            r_up = hand_up(kpts,"right",cfg.hand_offset)
            l_up = hand_up(kpts,"left",cfg.hand_offset)

            ax = int(kpts[12].x*W)
            ay = int(kpts[12].y*H)-12
            if ay<60: ay=int(kpts[28].y*H)+30

            put_info(frame,(ax,max(80,ay)), dist,height,r_up,l_up)

        # FPS
        now=time.time()
        fps=1/(now-prev_t)
        prev_t=now
        cv2.putText(frame,f"FPS: {fps:.1f}",(20,80),
            cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,150,0),2)

        cv2.imshow(win, frame)

        if (cv2.waitKey(1)&0xFF) in (ord('q'),ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




