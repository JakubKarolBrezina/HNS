# Computer Vision – YOLO Object Detection & MediaPipe Pose Estimation

This project demonstrates two computer vision applications:
1. **Object Detection using YOLO (You Only Look Once)**
2. **Human Pose Estimation using Google MediaPipe Pose Landmarker**

Both applications work with images, videos, or real-time camera input and are intended
for educational and experimental purposes.

---

## 1. Project Overview

### YOLO Object Detection
YOLO is used for real-time object detection. The application detects common objects
and people in images, videos, or webcam streams using pretrained models.

### MediaPipe Pose Estimation
MediaPipe Pose is used to detect and track human body landmarks in real time.
The application visualizes a full body skeleton and estimates basic physical attributes.

---

## 2. Features

### YOLO Object Detection
- object and person detection
- image, video, and webcam support
- real-time inference
- bounding boxes with class labels and confidence scores

### MediaPipe Pose Estimation
- detection of multiple people
- tracking of 33 human body landmarks
- full body skeleton visualization
- raised left and right hand detection
- approximate distance from camera estimation
- approximate person height estimation
- real-time FPS and average FPS display
- works in low-light conditions

---

## 3. System Requirements

### Software
- Python **3.9 or newer**
- pip (Python package manager)

### Required Python Libraries
- ultralytics
- mediapipe
- opencv-python
- numpy
- requests

---

## 4. Setup Instructions

### Step 1 – Check Python installation
```bash
python --version


