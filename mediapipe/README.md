# MediaPipe Pose Estimation

This project uses **Google MediaPipe Pose Landmarker** to detect and track
human body landmarks in real time from a webcam or video file.

The application provides real-time pose estimation and basic human posture analysis.

---

## Features

The application can:
- detect multiple people
- draw a full body skeleton using pose landmarks
- estimate distance from the camera
- estimate approximate person height
- detect raised left and right hand
- work in low-light conditions
- display real-time FPS and average FPS

---

## Requirements

- Python 3.9 or newer
- pip (Python package manager)

### Python libraries
- mediapipe
- opencv-python
- numpy
- requests

---

## Installation

Clone the repository and install required dependencies:

```bash
pip install mediapipe opencv-python numpy requests

