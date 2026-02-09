import cv2
import torch
from mmpose.apis import init_model, inference_topdown
from mmdet.apis import init_detector, inference_detector
from mmpose.visualization import PoseLocalVisualizer
from mmengine.structures import InstanceData

# ==============================
# NASTAVENIA
# ==============================
POSE_CONFIG = "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
POSE_CHECKPOINT = "td-hm_hrnet-w48_8xb32-210e_coco-256x192.pth"

DET_CONFIG = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco.py"
DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_nano_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

CAMERA_INDEX = 0   # 0 = interná kamera, 1 = externá USB kamera

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD MODELOV
# ==============================
pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)
det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE)

visualizer = PoseLocalVisualizer()
visualizer.set_dataset_meta(pose_model.dataset_meta)

# ==============================
# KAMERA
# ==============================
cap = cv2.VideoCapture(CAMERA_INDEX)

print(f"Používaná kamera: {CAMERA_INDEX}")
print("ESC = ukončiť")

# ==============================
# HLAVNÁ SLUČKA
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # DETEKCIA OSÔB
    det_result = inference_detector(det_model, frame)
    pred_instances = det_result.pred_instances
    persons = pred_instances[pred_instances.labels == 0]  # class 0 = person

    bboxes = persons.bboxes.cpu().numpy()

    pose_results = inference_topdown(pose_model, frame, bboxes)

    # Vizualizácia
    visualizer.add_datasample(
        "pose",
        frame,
        data_sample=pose_results,
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=False,
        show=False
    )

    output = visualizer.get_image()
    cv2.imshow("HRNet Pose Detection", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
