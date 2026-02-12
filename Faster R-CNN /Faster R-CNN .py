import cv2
import torch
import numpy as np
from torchvision import models
import torchvision.transforms as T

# ===============================
# Nastavenie zariadenia
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Používam:", device)

# ===============================
# Načítanie Faster R-CNN
# ===============================
model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device)
model.eval()

# ===============================
# Transformácia vstupu
# ===============================
transform = T.Compose([
    T.ToTensor()
])

# ===============================
# Kamera
# 0 = interná
# 1 = externá
# ===============================
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Chyba pri otváraní kamery")
    exit()

prev_center = None
movement_threshold = 20  # citlivosť pohybu (pixely)

print("Stlač 'q' pre ukončenie")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_tensor = transform(frame).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score > 0.8:  # 1 = osoba (COCO dataset)
            x1, y1, x2, y2 = box.astype(int)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            moved = False

            if prev_center is not None:
                distance = np.sqrt(
                    (center_x - prev_center[0])**2 +
                    (center_y - prev_center[1])**2
                )

                if distance > movement_threshold:
                    moved = True

            prev_center = (center_x, center_y)

            # Farba podľa pohybu
            if moved:
                color = (0, 0, 255)  # červená = pohyb
                text = "POHYB"
            else:
                color = (0, 255, 0)  # zelená = bez pohybu
                text = "STOJI"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    cv2.imshow("Faster R-CNN Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
