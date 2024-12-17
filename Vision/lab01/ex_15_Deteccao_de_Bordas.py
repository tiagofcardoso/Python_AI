import cv2
import numpy as np
import os

img = cv2.imread(os.path.join(os.path.dirname(__file__), 'Ovar.jpg'), cv2.IMREAD_GRAYSCALE)

# Detecção de Bordas com Canny
edges_canny = cv2.Canny(img, 100, 200)

# Detecção de Bordas com Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobelx, sobely)
sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)

# Detecção de Bordas com Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Exibindo as imagens
# cv2.imshow('nome_da_janela', imagem) abre uma janela com a imagem.
cv2.imshow('Original', img)
cv2.imshow('Canny', edges_canny)
cv2.imshow('Sobel Magnitude', sobel_magnitude)
cv2.imshow('Laplacian', laplacian)

# Aguarda uma tecla ser pressionada para fechar as janelas.
cv2.waitKey(0)
cv2.destroyAllWindows()