import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from fer import FER

# Inicializar a câmera
cap = cv2.VideoCapture(0)

# Inicializar o detector de rostos
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Carregar o modelo de reconhecimento facial
model = InceptionResnetV1(pretrained='vggface2').eval()

# Se você estiver usando uma GPU, mova o modelo para a GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Inicializar o detector de expressões faciais
expression_detector = FER(mtcnn=True)

# Função para detectar expressões faciais
def detect_expression(face):
    result = expression_detector.detect_emotions(face)
    if result:
        return result[0]['emotions']
    return None

# Loop para capturar frames da câmera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar rostos no frame
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = frame[y1:y2, x1:x2]
            emotions = detect_expression(face)
            if emotions:
                # Desenhar a caixa ao redor do rosto e exibir a emoção
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                emotion = max(emotions, key=emotions.get)
                cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()