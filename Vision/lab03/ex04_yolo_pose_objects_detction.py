import torch
import cv2
import mediapipe as mp

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(2)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = results.render()[0]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    if pose_results.pose_landmarks:
        frame_copy = frame.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            frame_copy, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        frame = frame_copy

    out.write(frame)
    cv2.imshow('YOLOv5 Detection and Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
