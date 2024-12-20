import cv2
import numpy as np
import os 

img_rgb = cv2.imread(os.path.join(os.path.dirname(__file__),'coins_euros.jpg'))
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(os.path.join(os.path.dirname(__file__), 'coin_20cents.jpg'), 0)

if template is None:
    raise FileNotFoundError("Template image 'coin_20cents.jpg' not found.")

# Obtém as dimensões do template.
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Define um limiar (threshold) para considerar uma correspondência como válida.
# Quanto mais próximo de 1, mais estrita a correspondência.
threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# guardar imagem resultante com as detecções marcadas.
# Save the resulting image with detections marked.
# Parameters: 'res_euros.jpg' is the filename, img_rgb is the image to be saved.
# Display the resulting image with detected matches in a window.
cv2.imshow('Detections', img_rgb)

cv2.imshow('Detecções', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()