import cv2
import os

# Define the path to the haarcascades directory
haarcascade_path = os.path.join(os.path.dirname(__file__), 'haarcascades')

# List all XML files in the haarcascades directory
xml_files = [f for f in os.listdir(haarcascade_path) if f.endswith('.xml')]
print("XML files in haarcascades directory:", xml_files)

# Load the face and eye cascade classifiers
faceCascade = cv2.CascadeClassifier(os.path.join(haarcascade_path, 'haarcascade_frontalface_default.xml'))
eyeCascade = cv2.CascadeClassifier(os.path.join(haarcascade_path, 'haarcascade_eye.xml'))

# Check if the cascade classifiers are loaded successfully
if faceCascade.empty() or eyeCascade.empty():
    print("Error loading cascade classifiers")
    exit()

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening video capture")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Start the time count
    tick_start = cv2.getTickCount()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define the region of interest (ROI) for eyes within the face
        roiGray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roiGray)

        eyeCounter = 0
        for ex, ey, ew, eh in eyes:
            if eyeCounter > 1:
                break
            # Draw a rectangle around the eyes
            cv2.rectangle(frame, (ex + x, ey + y), (ex + x + ew, ey + eh), (0, 255, 0), 2)
            eyeCounter += 1

    # End the time count
    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    # Print the time per frame (optional)
    print(f"Time per frame: {time_per_frame:.6f} seconds")

    # Display the frame with detected faces and eyes
    cv2.imshow('Face and Eye Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
