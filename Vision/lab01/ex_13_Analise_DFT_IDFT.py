import cv2
import numpy as np
import os

img = cv2.imread(os.path.join(os.path.dirname(__file__), 'Ovar.jpg'), cv2.IMREAD_GRAYSCALE)
dft_input = np.float32(img)
dft = cv2.dft(dft_input, flags=cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)

magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
magnitude_spectrum = 20 * np.log(magnitude + 1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
magnitude_spectrum = np.uint8(magnitude_spectrum)

# Exibe a imagem original e o espectro de magnitude.
cv2.imshow('Imagem Original', img)
cv2.imshow('Espectro de Magnitude', magnitude_spectrum)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Report imagem inicial
dft_ishift = np.fft.ifftshift(dft_shift)
img_back_complex = cv2.idft(dft_ishift)
img_back = cv2.magnitude(img_back_complex[:,:,0], img_back_complex[:,:,1])
img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back_normalized = np.uint8(img_back_normalized)

# Exibe a imagem reconstruída.
cv2.imshow('Imagem Reconstruida (IDFT)', img_back_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()