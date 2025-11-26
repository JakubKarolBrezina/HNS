import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import time
import math
import numpy as np
from ultralytics import YOLO
import ctypes

# ---------------- FULL TRUE FULLSCREEN ---------------- #
def force_fullscreen(window_name):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        ctypes.windll.user32.SetWindowLongW(hwnd, -16, 0x80000000)  # WS_POPUP
        ctypes.windll.user32.ShowWindow(hwnd, 3)                   # SW_MAXIMIZE

# --------------- CONFIG ---------------- #
REAL_SHOULDER_M = 0.42
FOV_DEG = 60
WINDOW_NAME = "YOLO11 POSE — FULLSCREEN"

# --------------- YOLO POSE --------------- #
# >>> Toto je dôležité – POUŽI POSE MODEL <<<
model = YOLO("../yolo11n-pose.pt")

# --------------- CAMERA --------------- #
cap = cv2.VideoCapture(0)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("✅ YOLO11 POSE FULLSCREEN — Press Q to quit")

prev = time.time()

# --------------- FUNKCIE --------------- #
def focal_len_px(img_w, fov_deg):
    return (img_w / 2) / math.tan(math.radians(fov_deg / 2))

def estimate_distance(shoulder_px, f_px):
    if shoulder_px < 5: return None
    return (f_px * REAL_SHOULDER_M) / shoulder_px

def estimate_height(box_h, shoulder_px):
    if shoulder_px < 5: return None
    return (box_h * (REAL_SHOULDER_M / shoulder_px))

# ---------------------------------------------- #
# -----------------   LOOP   -------------------- #
# ---------------------------------------------- #
while True:
    ret, frame = cap.read()
    if not ret: break

    force_fullscreen(WINDOW_NAME)

    H, W = frame.shape[:2]
    f_px = focal_len_px(W, FOV_DEG)

    results = model(frame, verbose=False)

    for r in results:
        for box, keypoints in zip(r.boxes, r.keypoints):
            cls = int(box.cls[0])
            if cls != 0:
                continue

            # --- BOUNDING BOX ---
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            box_w = x2 - x1
            box_h = y2 - y1

            # --- SHOULDER WIDTH (approx) ---
            shoulder_px = int(box_w * 0.40)

            # --- ESTIMATIONS ---
            dist = estimate_distance(shoulder_px, f_px)
            height = estimate_height(box_h, shoulder_px)

            # === DRAW BOX ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.putText(frame, f"Person {conf*100:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if height:
                cv2.putText(frame, f"H: {height:.2f} m", (x1, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            if dist:
                cv2.putText(frame, f"D: {dist:.2f} m", (x1, y1 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # --- SKELETON KEYPOINTS ---
            kpts = keypoints.xy[0].cpu().numpy()

            for (x, y) in kpts:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw lines (official COCO order)
            skeleton = [
                (5, 7), (7, 9),
                (6, 8), (8, 10),
                (5, 6),
                (11, 12),
                (5, 11), (6, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16)
            ]

            for a, b in skeleton:
                xa, ya = kpts[a]
                xb, yb = kpts[b]
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 2)

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
        break

cap.release()
cv2.destroyAllWindows()


