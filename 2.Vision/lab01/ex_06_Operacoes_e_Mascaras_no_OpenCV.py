import cv2
import numpy as np
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')
bot = os.path.join(os.path.dirname(__file__), 'bot.png')
bot_background = os.path.join(os.path.dirname(__file__), 'bot_background.png')

x=np.uint8([[250]])
y=np.uint8([[10]])
print(f'Numpy add {x+y}')   # 4
print(f'CvAdd add {cv2.add(x,y)}') #255

img1 = cv2.imread(ovar)
img2 = cv2.imread(bot)
img3 = cv2.imread(bot_background)
final =  cv2.addWeighted(img2,0.7, img3, 0.3,0)

# Definir a ROI na imagem de fundo baseada no tamanho do robô
rows, cols, channels=img2.shape
roi = img1[0:rows, 0:cols]

# Converter a imagem do robô para escala de cinza e criar máscara
img2gray= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask=cv2.threshold(img2gray,245, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Remover a área do robô da ROI do fundo
img1_bg = cv2.bitwise_and(roi,roi, mask=mask)

# Obter apenas a área do robô
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

# Combinar as áreas do fundo e do robô
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('cv2.add: dst', dst)
cv2.imshow('cv2.addWeighted final', final)
cv2.waitKey()
cv2.destroyAllWindows()