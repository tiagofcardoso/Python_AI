import cv2

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        break

    # Exibir o frame
    cv2.imshow('Frame', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar janelas
cap.release()
cv2.destroyAllWindows()
