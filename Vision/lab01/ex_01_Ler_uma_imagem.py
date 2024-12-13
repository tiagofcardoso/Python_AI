import cv2
import os

ovar = os.path.join(os.path.dirname(__file__), 'Ovar.jpg')
img_gray  = cv2.imread(ovar,0)
img_bgr  = cv2.imread(ovar,cv2.IMREAD_COLOR) # 1

if img_gray is None:
    print("Erro: Não foi possível carregar a imagem. Verifique o nome do ficheiro e a localização.")
    exit()
    
cv2.imshow('imagem Greyscale', img_gray)
cv2.imshow('imagem Cor', img_bgr)
cv2.waitKey()
cv2.destroyAllWindows()