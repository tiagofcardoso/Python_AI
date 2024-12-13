import cv2
import numpy as np
import os

bot = os.path.join(os.path.dirname(__file__), 'bot.png')

img = cv2.imread(bot, cv2.IMREAD_COLOR)

kernel = np.ones((5,5), np.uint8)

print(kernel)

erosion =  cv2.erode(img,kernel, iterations=1)
dilation =  cv2.dilate(img,kernel, iterations=1)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN ,kernel)
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE ,kernel)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT ,kernel)
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT ,kernel)
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT ,kernel)

# Exibe as imagens resultantes em janelas separadas.
cv2.imshow('Original', img)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Gradient', gradient)
cv2.imshow('Top-Hat', tophat)
cv2.imshow('Black-Hat', blackhat)
cv2.waitKey()
cv2.destroyAllWindows()
