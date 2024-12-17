import numpy as np
import cv2

cap = cv2.VideoCapture(0)


ret, frame = cap.read()
if not ret:
    print("Não foi possível ler o primeiro frame da webcam.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

rows, cols = frame.shape[:2]
windowWidth = 150
windowHeight = 200
windowCol = int((cols - windowWidth) / 2)
windowRow = int((rows - windowHeight) / 2)
window = (windowCol, windowRow, windowWidth, windowHeight)

roi = frame[windowRow:windowRow + windowHeight,
            windowCol:windowCol + windowWidth]
roiHsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

lowLimit = np.array((0., 60., 32.))
highLimit = np.array((180., 255., 255.))
mask = cv2.inRange(roiHsv, lowLimit, highLimit)

roiHist = cv2.calcHist([roiHsv], [0], mask, [180], [0, 180])
cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

terminationCriteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Início da contagem de tempo
    tick_start = cv2.getTickCount()

    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backprojectedFrame = cv2.calcBackProject([frameHsv], [0], roiHist, [0, 180],
                                             1)
    mask = cv2.inRange(frameHsv, lowLimit, highLimit)
    backprojectedFrame &= mask

    # Aplicar CamShift
    ret, window = cv2.CamShift(backprojectedFrame, window, terminationCriteria)

    # Desenhar o polígono resultante do CamShift
    points = cv2.boxPoints(ret)
    points = np.int0(points)
    frame = cv2.polylines(frame, [points], True, 255, 2)

    # Fim da contagem de tempo
    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    # Imprimir tempo por frame (opcional)
    print(f"Tempo por frame: {time_per_frame:.6f} segundos")

    cv2.imshow('camshift', frame)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

cap.release()   
cv2.destroyAllWindows()