import cv2
import torch
import numpy as np
from torchvision import transforms

print("Načítavam HRNet model...")

model = torch.hub.load(
    'HRNet/HRNet-Human-Pose-Estimation',
    'pose_hrnet_w32',
    pretrained=True
)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model načítaný.")
print("Používané zariadenie:", device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera sa nepodarilo otvoriť.")
    exit()

print("Kamera otvorená.")
print("ESC = ukončiť")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame sa nepodarilo načítať.")
        break

    cv2.putText(frame, "HRNet bezi...", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    input_img = transform(frame)
    input_img = input_img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_img)

    heatmaps = output.cpu().numpy()

    keypoints = []
    for i in range(heatmaps.shape[1]):
        hm = heatmaps[0, i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        keypoints.append((x, y))

    h, w, _ = frame.shape
    sx = w / 192
    sy = h / 256

    for x, y in keypoints:
        cx = int(x * sx)
        cy = int(y * sy)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("HRNet Pose", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
