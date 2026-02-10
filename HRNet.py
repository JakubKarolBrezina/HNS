import cv2
import numpy as np
import onnxruntime as ort

# =========================
# NASTAVENIA
# =========================
MODEL_PATH = "hrnet_w32_coco_256x192.onnx"
CAMERA_INDEX = 0   # 0 = interná, 1 = externá
INPUT_SIZE = (192, 256)  # (W, H) podľa HRNet

# =========================
# LOAD MODELU
# =========================
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

# =========================
# KAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

def preprocess(frame):
    img = cv2.resize(frame, INPUT_SIZE)
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def get_keypoints(heatmaps):
    keypoints = []
    for i in range(heatmaps.shape[1]):
        hm = heatmaps[0, i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints.append((x, y))
    return keypoints

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)
    outputs = session.run(None, {input_name: inp})

    heatmaps = outputs[0]
    keypoints = get_keypoints(heatmaps)

    h, w, _ = frame.shape
    sx = w / INPUT_SIZE[0]
    sy = h / INPUT_SIZE[1]

    for x, y in keypoints:
        cx = int(x * sx)
        cy = int(y * sy)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.imshow("HRNet ONNX Pose", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
