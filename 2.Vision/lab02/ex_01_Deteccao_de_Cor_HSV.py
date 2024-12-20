import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:

    # Ler cada frame da webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir o intervalo de cor azul no espaço HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Criar a máscara
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Extrair apenas a parte azul da imagem
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Exibir as janelas
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Mascara', mask)
    cv2.imshow('Resultado', res)

    # Pressione ESC para sair
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()