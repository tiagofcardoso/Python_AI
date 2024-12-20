import cv2
from matplotlib import pyplot as plt

# Carregar a imagem em tons de cinza
img  = cv2.imread('Ovar.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar a imagem usando matplotlib
plt.imshow(img, cmap='gray', interpolation='bicubic')

# Ocultar os valores dos eixos
plt.xticks([])
plt.yticks([])

# Exibir o gráfico
plt.show()