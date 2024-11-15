import cv2
import mediapipe as mp

# Inicializar mediapipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Função para reconhecer gestos básicos
def recognize_gesture(hand_landmarks):
    thumb_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    index_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_is_open = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    if thumb_is_open and index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "Open Hand"
    elif not thumb_is_open and index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "Peace"
    elif not thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Pointing"
    elif not thumb_is_open and not index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "F***You"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "Hang Loose"
    elif thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and pinky_is_open:
        return "Rock n Roll"
    elif thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Positive"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Negative"
    elif not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "Closed Fist"
    else:
        return "Unknown Gesture"

# Capturar vídeo da câmera do laptop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processar a imagem para detectar mãos
    results = hands.process(image)

    # Converter a imagem de volta para BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenhar as marcações das mãos e reconhecer gestos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)
            cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibir o frame com as detecções
    cv2.imshow('Hand Gesture Recognition', image)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar janelas
cap.release()
cv2.destroyAllWindows()
hands.close()