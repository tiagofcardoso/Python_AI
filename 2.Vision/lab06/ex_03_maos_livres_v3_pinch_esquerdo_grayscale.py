import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def vector(a, b):
    """Retorna o vetor AB = b - a."""
    return (b[0] - a[0], b[1] - a[1])

def dot(v1, v2):
    """Produto escalar de v1 e v2."""
    return v1[0]*v2[0] + v1[1]*v2[1]

def magnitude(v):
    """Magnitude (comprimento) do vetor."""
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

def angle_3points(a, b, c):
    """
    Calcula o ângulo em graus entre os vetores BA e BC,
    isto é, o ângulo no ponto B dado por A-B-C.
    a, b, c são tuplas (x, y).
    """
    # Vetores BA e BC
    ba = vector(b, a)
    bc = vector(b, c)

    # Produto escalar e magnitudes
    dot_prod = dot(ba, bc)
    mag_ba = magnitude(ba)
    mag_bc = magnitude(bc)

    # Evitar divisão por zero
    if mag_ba * mag_bc == 0:
        return 0.0

    # Ângulo em graus
    cos_angle = dot_prod / (mag_ba * mag_bc)
    # Ajustar para evitar problemas numéricos fora do intervalo [-1, 1]
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def finger_is_extended(hand_landmarks, finger_name):
    """
    Verifica se um determinado dedo (finger_name) está esticado baseado no ângulo.
    finger_name pode ser: "thumb", "index", "middle", "ring", "pinky".
    Retorna True (esticado) ou False (dobrado).
    """
    if finger_name == "thumb":
        mcp_idx, pip_idx, tip_idx = 2, 3, 4  # MCP(2), IP(3), TIP(4)
    elif finger_name == "index":
        mcp_idx, pip_idx, tip_idx = 5, 6, 8
    elif finger_name == "middle":
        mcp_idx, pip_idx, tip_idx = 9, 10, 12
    elif finger_name == "ring":
        mcp_idx, pip_idx, tip_idx = 13, 14, 16
    elif finger_name == "pinky":
        mcp_idx, pip_idx, tip_idx = 17, 18, 20
    else:
        return False

    mcp = hand_landmarks[mcp_idx]
    pip_ = hand_landmarks[pip_idx]
    tip = hand_landmarks[tip_idx]

    ang = angle_3points(mcp, pip_, tip)

    # Definir um limiar para dizer se está "esticado"
    if ang > 160:
        return True
    else:
        return False

def detect_gesture(lm_list):
    """
    Detecta gesto a partir do estado (esticado ou não) de cada dedo.
    Gera um "mapa" dos dedos esticados para comparar com gestos pré-definidos.
    """
    thumb_extended  = finger_is_extended(lm_list, "thumb")
    index_extended  = finger_is_extended(lm_list, "index")
    middle_extended = finger_is_extended(lm_list, "middle")
    ring_extended   = finger_is_extended(lm_list, "ring")
    pinky_extended  = finger_is_extended(lm_list, "pinky")

    extended_fingers = [
        thumb_extended,
        index_extended,
        middle_extended,
        ring_extended,
        pinky_extended
    ]
    extended_count = sum(extended_fingers)

    thumb_tip = lm_list[4]
    index_tip = lm_list[8]
    dist_thumb_index = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])

    # Exemplos de gestos:
    if extended_count == 0:
        return "FIST"
    if extended_count == 5:
        return "OPEN"
    if dist_thumb_index < 30:
        # Se middle, ring e pinky também estiverem esticados, OK clássico
        if middle_extended and ring_extended and pinky_extended:
            return "OK"
        else:
            return "PINCH"
    if index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        return "POINT"
    if thumb_extended and (not index_extended) and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        return "THUMBS_UP"
    if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
        return "PEACE"
    if index_extended and (not middle_extended) and (not ring_extended) and pinky_extended:
        return "ROCK"

    return "DESCONHECIDO"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Variável para controlar o filtro a cada quadro
    # Pode ter valores "NONE", "GRAY" ou "BGR"
    current_filter = "NONE"

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Identificar mão esquerda ou direita
            handed_label = results.multi_handedness[hand_idx].classification[0].label
            if handed_label == "Left":
                mao_str = "Mao Esquerda"
            elif handed_label == "Right":
                mao_str = "Mao Direita"
            else:
                mao_str = "Mao Desconhecida"

            # Extrair (x, y) de cada landmark
            lm_list = []
            for lm in hand_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                lm_list.append((px, py))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar o gesto
            gesto = detect_gesture(lm_list)

            # Se for PINCH, alteramos o filtro de acordo com a mão
            if gesto == "PINCH":
                if mao_str == "Mao Esquerda":
                    current_filter = "GRAY"
                else:
                    current_filter = "BGR"

            # Exibir texto
            texto = f"{mao_str}: {gesto}"
            cv2.putText(
                frame,
                texto,
                (10, 60 + hand_idx * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )

    # Aplica o filtro, se necessário:
    if current_filter == "GRAY":
        # Converte para grayscale e então para 3 canais (para manter compatível com o imshow)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif current_filter == "BGR":
        # Mantém como está (BGR), pois já está em BGR.
        # Se quiser algum outro efeito, coloque aqui.
        pass

    cv2.imshow("Detecao de Maos e Gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
