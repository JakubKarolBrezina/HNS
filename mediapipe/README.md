# MediaPipe Pose Estimation

This project performs real-time human pose estimation using
**Google MediaPipe Pose Landmarker**.  
It detects and tracks human body landmarks from a webcam or a video file.

---

## 1. Project Description

The application detects human body pose in real time and visualizes
a full body skeleton based on detected landmarks.
It also provides basic posture-related information such as distance,
height estimation, and hand position detection.

---

## 2. Features


- detection of multiple people
- tracking of 33 human body landmarks
- real-time full body skeleton visualization
- raised left and right hand detection
- approximate distance from the camera estimation
- approximate person height estimation
- real-time FPS and average FPS display
- works in low-light conditions

---

## 3. System Requirements

### Software
- Python **3.9 or newer**
- pip (Python package manager)

### Required Python Libraries
- mediapipe
- opencv-python
- numpy
- requests

---

## 4. Setup Instructions

### Step 1 – Check Python installation
```bash
python --version



