import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

gray_img = cv2.imread(os.path.join(os.path.dirname(__file__), 'Ovar.jpg'), cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', gray_img)

imS = cv2.resize(gray_img, (1024,768))
# cv2.imshow('imS', imS)

# Calcula o histograma com 256 bins para comparação
hist_64 = cv2.calcHist([gray_img], [0], None, [64], [0,256])
hist_256=cv2.calcHist([imS],[0], None, [256], [0,256])
print('hist_256',hist_256)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(gray_img.ravel(), 256, [0,256])
plt.title('Histograma (256 bins)')
plt.xlabel('Intensidade')
plt.ylabel('Frequência')

# Calcula o histograma com 64 bins para comparação
hist_64 = cv2.calcHist([gray_img], [0], None, [64], [0,256])
print('hist_64',hist_64)

plt.subplot(1,2,2)
plt.hist(gray_img.ravel(), 64, [0,256])
plt.title('Histograma (64 bins)')
plt.xlabel('Intensidade')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

plt.show()
cv2.waitKey()
cv2.destroyAllWindows()