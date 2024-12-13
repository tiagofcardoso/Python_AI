import cv2
import matplotlib.pyplot as plt
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')

# Carregar a imagem Ovar.jpg
img =  cv2.imread(ovar)
if img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Exibir a imagem diretamente no Matplotlib (BGR interpretado como RGB)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(img)  # Cores incorretas
plt.title('BGR exibido diretamente')
plt.axis('off')

# Converter BGR -> RGB para exibição correta
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,3,2)
plt.imshow(img_rgb)
plt.title('Imagem em RGB')
plt.axis('off')

# Converter BGR -> HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.subplot(1,3,3)
plt.imshow(img_hsv)
plt.title('Imagem em HSV (interpretada como RGB)')
plt.axis('off')

# Mostrar as 3 imagens em um único plot
plt.tight_layout()
plt.show()

#Separar os canais H,S,V
h,s,v = cv2.split(img_hsv)

plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.imshow(h, cmap='gray')
plt.title('Canal H')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(s, cmap='gray')
plt.title('Canal S')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(v, cmap='gray')
plt.title('Canal V')
plt.axis('off')

plt.tight_layout()
plt.show()