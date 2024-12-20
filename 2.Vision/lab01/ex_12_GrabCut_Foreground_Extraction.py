import cv2
import numpy as np
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')
img = cv2.imread(ovar)

mask = np.zeros(img.shape[:2], np.uint8)

# Auxiliares
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)

# Define um retângulo que engloba a área aproximada do objeto a ser extraído.
rect = (50, 50, 450, 290)

# Aplica o algoritmo GrabCut.
# Após o GrabCut, a máscara contém a zona a alterar
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img_result = img * mask2[:, :, np.newaxis]

# Exibe a imagem original e a imagem resultante.
cv2.imshow('Original', cv2.imread(ovar))
cv2.imshow('Foreground Extraido', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
