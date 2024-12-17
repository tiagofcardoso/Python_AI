import numpy as np
import cv2
import os

# Load the video
cap = cv2.VideoCapture(0)

# Parameters for ShiTomasi corner detection
ret, frame = cap.read()
if not ret:
    print("Error reading video")
    cap.release()
    cap.destroyAllWindows()
    exit()

rows, cols = frame.shape[:2]

windWidth = 150
windowHeight = 200
windowCol = int((cols - windWidth) / 2)
windowRow = int((rows - windowHeight) / 2)
window = (windowCol, windowRow, windWidth, windowHeight)

roi = frame[windowRow:windowRow+windowHeight, windowCol:windowCol+windWidth]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lowLimit = np.array([0, 60, 32])
highLimit = np.array([180, 255, 255])
mask = cv2.inRange(roi_hsv, lowLimit, highLimit)

roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

termCriteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    tick_start = cv2.getTickCount()
    
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backprojecteFrame = cv2.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)

    mask = cv2.inRange(frame_hsv, lowLimit, highLimit)
    backprojecteFrame &= mask

    ret, window = cv2.meanShift(backprojecteFrame, window, termCriteria)

    windowCol, windowRow = window[:2]
    cv2.rectangle(frame, (windowCol, windowRow), (windowCol+windWidth, windowRow+windowHeight), (0, 255, 0), 2)

    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    print("Time per frame: %.2f [ms]" % (time_per_frame * 1000))

    cv2.imshow('MeanShift Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
