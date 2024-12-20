import cv2
import os

bot = os.path.join(os.path.dirname(__file__), 'bot.png')

img = cv2.imread(bot)

# Aplica o filtro blocks 5x5
blur = cv2.blur(img, (5,5))

# Aplica o filtro Gaussiano
gaussian_blur = cv2.GaussianBlur(img, (5,5),0)

# Aplica o filtro Gaussiano
median =  cv2.medianBlur (img, 5)

bilateral =  cv2.bilateralFilter(img, 9, 75,75)

# Exibe as imagens em janelas separadas
cv2.imshow('Original', img)
cv2.imshow('Blur (Media)', blur)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median)
cv2.imshow('Bilateral Filter', bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()
