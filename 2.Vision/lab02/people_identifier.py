import cv2
import os
import numpy as np

# Load pre-trained models for object detection
def initialize_model():
    net = cv2.dnn.readNetFromCaffe(os.path.join(os.path.dirname(__file__), 'deploy.prototxt'), os.path.join(os.path.dirname(__file__), 'res10_300x300_ssd_iter_140000.caffemodel'))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

# Load video
video = os.path.join(os.path.dirname(__file__), 'vtest.avi')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
if not out.isOpened():
    print("Error: Could not open video writer.")
    cap.release()
    cv2.destroyAllWindows()
    exit()
QUIT_KEY = ord('q')
LABELS = {
    15: 'Person',
    6: 'Bus',
    14: 'Motorbike',
    2: 'Bicycle',
    8: 'Cat',
    12: 'Dog',
    13: 'Horse'
}

# Initialize the model
net = initialize_model()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw rectangles around detected objects
                label = int(detections[0, 0, i, 1])
                if label in LABELS:
                    if label == 15:  # Person
                        color = (0, 0, 255)  # Red
                    elif label in [7, 6, 14, 2]:  # Car, Bus, Motorbike, Bicycle
                        color = (255, 0, 0)  # Blue
                    elif label in [8, 12, 13]:  # Animals (Cat, Dog, Horse)
                        color = (0, 255, 0)  # Green
                else:
                    color = (255, 255, 255)  # White for other objects

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    out.release()
    cv2.destroyAllWindows()