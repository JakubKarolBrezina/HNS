#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import math
import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

import cv2
import numpy as np
from ultralytics import YOLO


# ==============================
# ---------- KONFIG -----------
# ==============================

@dataclass
class AppConfig:
    model_path: str = "../yolo11n-pose.pt"
    cam_index: int = 0
    fov_deg: float = 60.0
    real_shoulder_m: float = 0.42
    conf_thr: float = 0.6
    safe_zone_m: float = 1.0
    warn_zone_m: float = 2.0
    draw_thick: int = 2
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


# ==============================
# ---------- KEYPOINTS ---------
# ==============================

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


# ==============================
# --------- UTIL FUNKCIE -------
# ==============================

def focal_len_px(img_w, fov_deg):
    return (img_w/2) / math.tan(math.radians(fov_deg/2))


def shoulder_px(kpts_xy):
    L = kpts_xy[KP["lsho"], 0]
    R = kpts_xy[KP["rsho"], 0]
    if np.isnan(L) or np.isnan(R):
        return 0
    return abs(R - L)


def estimate_distance_m(shoulder_px_val, f_px, real_shoulder_m):
    if shoulder_px_val <= 5:
        return None
    return (f_px * real_shoulder_m) / shoulder_px_val


def estimate_height_m(k, s_px, real_shoulder_m):
    if s_px <= 5:
        return None

    top = k[KP["nose"], 1]
    ankles = [k[KP["lank"], 1], k[KP["rank"], 1]]
    ankles = [x for x in ankles if not np.isnan(x)]
    if not ankles:
        return None

    px_h = max(ankles) - top
    if px_h <= 0:
        return None

    return px_h * (real_shoulder_m / s_px)


# ==============================
# ---- HAND-UP (3-LEVEL) -------
# ==============================

def hand_up_both(k):
    try:
        r_sh = k[KP["rsho"], 1]
        r_wr = k[KP["rwri"], 1]
        l_sh = k[KP["lsho"], 1]
        l_wr = k[KP["lwri"], 1]

        if any(np.isnan(v) for v in [r_sh, r_wr, l_sh, l_wr]):
            return None

        R = r_wr < r_sh - 40
        L = l_wr < l_sh - 40

        if R and L:
            return "both"
        if R:
            return "right"
        if L:
            return "left"
        return None

    except:
        return None


def torso_tilt_deg(k):
    try:
        sx = (k[5,0] + k[6,0]) / 2
        sy = (k[5,1] + k[6,1]) / 2
        hx = (k[11,0] + k[12,0]) / 2
        hy = (k[11,1] + k[12,1]) / 2
        return abs(math.degrees(math.atan2(sx - hx, sy - hy)))
    except:
        return None


def decide_safety(dist_m, cfg):
    if dist_m is None:
        return Safety.GO
    if dist_m < cfg.safe_zone_m:
        return Safety.STOP
    if dist_m < cfg.warn_zone_m:
        return Safety.SLOW
    return Safety.GO


# ==============================
# --------- FALL FILTER --------
# ==============================

def update_fall_state(ts, height_m, tilt, bbox, now):
    if height_m is None or tilt is None:
        ts.fall_sus_frames = 0
        ts.prev_height_m = height_m
        ts.prev_time = now
        return False

    x1,y1,x2,y2 = bbox
    w = max(1,x2-x1)
    h = max(1,y2-y1)
    horiz = w/h

    # baseline calibrácia
    if tilt < 30:
        if ts.standing_height_m is None:
            ts.standing_height_m = height_m
        else:
            ts.standing_height_m = ts.standing_height_m*0.9 + height_m*0.1

    if ts.standing_height_m is None:
        ts.prev_height_m = height_m
        ts.prev_time = now
        return False

    too_low = height_m < 0.6 * ts.standing_height_m
    very_low = height_m < 0.45 * ts.standing_height_m

    big_drop = False
    if ts.prev_height_m is not None:
        dt = now - ts.prev_time
        if dt>0:
            rel = (ts.prev_height_m - height_m) / max(ts.prev_height_m,1e-6)
            if dt<0.7 and rel>0.35:
                big_drop = True

    suspicious = (
        tilt > 75 and
        horiz > 1.3 and
        (very_low or (too_low and big_drop))
    )

    ts.fall_sus_frames = ts.fall_sus_frames+1 if suspicious else max(0, ts.fall_sus_frames-1)

    if ts.fall_sus_frames >= 10 and now - ts.last_fall_time > 3:
        ts.last_fall_time = now
        ts.fall_sus_frames = 0
        return True

    ts.prev_height_m = height_m
    ts.prev_time = now
    return False


# ==============================
# ------------ MQTT ------------
# ==============================

class MqttClient:
    def __init__(self, host, port):
        self.client = None
        try:
            import paho.mqtt.client as mqtt
            if host:
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                c.connect(host, port, 60)
                c.loop_start()
                self.client = c
        except:
            print("⚠ MQTT disabled")

    def publish(self, t, msg):
        if self.client:
            self.client.publish(t, msg)

    def close(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# ==============================
# ----------- DRAW -------------
# ==============================

def draw_pose(f, k, thick=2):
    for x,y in k:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(f,(int(x),int(y)),4,(0,0,255),-1)

    for a,b in SKELETON:
        xa,ya = k[a]
        xb,yb = k[b]
        if not np.isnan(xa) and not np.isnan(ya) and not np.isnan(xb) and not np.isnan(yb):
            cv2.line(f,(int(xa),int(ya)),(int(xb),int(yb)),(255,255,255),thick)


# ==============================
# ------------- MAIN -----------
# ==============================

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cfg = AppConfig(cam_index=args.cam)
    mqtt = MqttClient(cfg.mqtt_host, cfg.mqtt_port)

    model = YOLO(cfg.model_path)

    cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened():
        print("❌ Kamera sa nedala otvoriť.")
        return

    cap.set(3,1920)
    cap.set(4,1080)

    win="YOLO-FULLSCREEN"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    track_states: Dict[int,TrackState] = {}

    prev_t = time.time()

    print("SYSTEM ONLINE – stlač Q pre ukončenie.\n")

    while True:

        ok, frame = cap.read()

        if not ok or frame is None:
            dbg = np.zeros((720,1280,3),dtype=np.uint8)
            cv2.putText(dbg,"KAMERA NEDAVA OBRAZ",(50,360),
                        cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,255),3)
            cv2.imshow(win, dbg)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        H,W = frame.shape[:2]
        f_px = focal_len_px(W, cfg.fov_deg)

        results = model.track(frame, persist=True, conf=cfg.conf_thr, classes=[0], verbose=False)

        now = time.time()
        fps = 1.0 / max(now-prev_t,1e-6)
        prev_t = now

        if results is None or len(results)==0:
            cv2.putText(frame, "NO DETECTIONS", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow(win,frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        r = results[0]
        boxes = r.boxes
        kpts = r.keypoints

        if boxes is not None and kpts is not None:

            for i in range(len(boxes)):
                if int(boxes.cls[i].item()) != 0:
                    continue

                track_id = int(boxes.id[i].item()) if boxes.id is not None else i+1

                x1,y1,x2,y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                kxy = kpts.xy[i].cpu().numpy()
                kxy = np.where(np.isfinite(kxy), kxy, np.nan)

                draw_pose(frame, kxy, 2)

                s_px = shoulder_px(kxy)
                dist = estimate_distance_m(s_px, f_px, cfg.real_shoulder_m)
                height = estimate_height_m(kxy, s_px, cfg.real_shoulder_m)
                tilt = torso_tilt_deg(kxy)

                # -------------------------------
                #   🟩 HAND-UP 3-level detection
                # -------------------------------
                hand = hand_up_both(kxy)

                if hand == "right":
                    cv2.putText(frame,"RIGHT HAND UP",(x1,y1-60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)
                    mqtt.publish(cfg.topic_event, f"{track_id}:RIGHT_UP")

                elif hand == "left":
                    cv2.putText(frame,"LEFT HAND UP",(x1,y1-60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)
                    mqtt.publish(cfg.topic_event, f"{track_id}:LEFT_UP")

                elif hand == "both":
                    cv2.putText(frame,"BOTH HANDS UP",(x1,y1-60),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                    mqtt.publish(cfg.topic_event, f"{track_id}:BOTH_UP")

                state = decide_safety(dist, cfg)

                ts = track_states.setdefault(track_id, TrackState())

                fall = update_fall_state(ts, height, tilt, (x1,y1,x2,y2), now)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                info = f"ID:{track_id}"
                if height: info+=f" H:{height:.2f}"
                if dist: info+=f" D:{dist:.2f}"
                if tilt: info+=f" tilt:{tilt:.0f}"

                cv2.putText(frame, info, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

                if fall:
                    cv2.putText(frame,"FALL DETECTED!",(x1,y1-40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)
                    mqtt.publish(cfg.topic_event, f"{track_id}:FALL")

                mqtt.publish(cfg.topic_state, f"{track_id}:{state.value}")

        cv2.putText(frame,f"FPS:{fps:.1f}",(20,80),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,180,255),2)

        cv2.imshow(win,frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mqtt.close()


if __name__ == "__main__":
    main()




