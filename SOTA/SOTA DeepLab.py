import cv2
import torch
import torchvision.transforms as T
import numpy as np
from torchvision import models

# ===============================
# Nastavenie zariadenia (GPU/CPU)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Používam zariadenie:", device)

# ===============================
# Načítanie SOTA DeepLabV3+
# ===============================
model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
model.to(device)
model.eval()

# ===============================
# Transformácia vstupu
# ===============================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((480, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ===============================
# Výber kamery
# 0 = interná
# 1 = externá
# ===============================
camera_index = 0  # zmeň na 1 ak chceš externú
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Chyba pri otváraní kamery.")
    exit()

prev_mask = None

print("Stlač 'q' pre ukončenie.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Trieda 15 = osoba (COCO dataset)
    person_mask = (output_predictions == 15).astype(np.uint8)

    person_mask = cv2.resize(person_mask, (frame.shape[1], frame.shape[0]))

    motion_mask = np.zeros_like(person_mask)

    if prev_mask is not None:
        diff = cv2.absdiff(person_mask, prev_mask)
        _, motion_mask = cv2.threshold(diff, 0.1, 1, cv2.THRESH_BINARY)

    prev_mask = person_mask.copy()

    # Vytvor červený overlay na pohyb
    overlay = frame.copy()
    overlay[motion_mask == 1] = [0, 0, 255]

    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    cv2.imshow("DeepLabV3+ Motion Detection", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
