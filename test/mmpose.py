import cv2
from mmpose.apis import MMPoseInferencer

inferencer = MMPoseInferencer('rtmpose-s')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = inferencer(frame, return_vis=True)
    vis = result['visualization'][0]

    cv2.imshow("RTMPose Tracking", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


