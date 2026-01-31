# MediaPipe Pose Estimation

## What does it do
This project uses Google MediaPipe Pose Landmarker to detect and track
human body landmarks in real time from a camera or video file.

The application can:
- detect multiple people
- draw a full body skeleton
- estimate distance from the camera
- estimate approximate person height
- detect raised left and right hand
- work in low-light conditions
- display real-time FPS and average FPS

---

## Setup

### Requirements
- Python 3.9+
- MediaPipe
- OpenCV
- NumPy
- Requests

### Install dependencies
```bash
pip install mediapipe opencv-python numpy requests
