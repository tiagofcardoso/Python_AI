import cv2
import numpy as np 
import os
img = cv2.imread(os.path.join(os.path.dirname(__file__), 'bot.png'))

# +--------------------------+
# Segmentação por Watershed
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#   Limpeza - Remover ruídos pequenos com abertura (erosão seguida de dilatação).
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)
watershed_result = img.copy()
watershed_result[markers == -1] = [0, 0, 255]


# +-------------------------------+
# Detecção de Linhas (Hough Lines)
gray_for_lines = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    aplica-se canny mas para prepara para Hough
edges = cv2.Canny(gray_for_lines, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
hough_lines_result = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Desenha a linha na imagem com cor verde e espessura 2.
        cv2.line(hough_lines_result, (x1,y1), (x2,y2), (0,255,0), 2)

# +-----------------------------------+
# Detecção de Círculos (Hough Circles)
#     Converter para escala de cinza e suavizar com mediana.
gray_for_circles = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_for_circles = cv2.medianBlur(gray_for_circles, 5)
circles = cv2.HoughCircles(gray_for_circles, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=30, minRadius=0, maxRadius=0)

hough_circles_result = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0,:]:
        # Desenha o círculo detectado em azul.
        cv2.circle(hough_circles_result, (c[0], c[1]), c[2], (255,0,0), 2)
        # Desenha o centro do círculo em vermelho.
        cv2.circle(hough_circles_result, (c[0], c[1]), 2, (0,0,255), 3)

# Exibir resultados
cv2.imshow('Original', img)
cv2.imshow('Watershed Result', watershed_result)
cv2.imshow('Hough Lines', hough_lines_result)
cv2.imshow('Hough Circles', hough_circles_result)

cv2.waitKey(0)
cv2.destroyAllWindows()