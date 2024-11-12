import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load an image from a local file
# Substitua pelo caminho da sua imagem
img_path = '/home/tiagocardoso/Downloads/zidane.jpg'
img = cv2.imread(img_path)

# Perform inference
results = model(img)

# Print results
results.print()

# Process the results
results.save()  # Save the results to disk (optional)
results.show()  # Display the results in a window (optional)

# Extract bounding boxes and labels
boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
labels = results.names  # Class labels

# Draw bounding boxes on the image
for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    label = f"{labels[int(cls)]}: {conf:.2f}"
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, label, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image with bounding boxes
cv2.imshow('YOLOv5 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
