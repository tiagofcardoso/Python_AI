import cv2
import mediapipe as mp
import math
from datetime import datetime  # Para gerar data/hora

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
    ba = vector(b, a)
    bc = vector(b, c)

    dot_prod = dot(ba, bc)
    mag_ba = magnitude(ba)
    mag_bc = magnitude(bc)

    if mag_ba * mag_bc == 0:
        return 0.0

    cos_angle = dot_prod / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def finger_is_extended(hand_landmarks, finger_name):
    """
    Verifica se um determinado dedo está esticado baseado no ângulo.
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
    return ang > 160

def detect_gesture(lm_list):
    """
    Detecta gesto a partir do estado (esticado ou não) de cada dedo.
    Retorna uma string representando o gesto.
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
    dist_thumb_index = math.hypot(
        thumb_tip[0] - index_tip[0],
        thumb_tip[1] - index_tip[1]
    )

    if extended_count == 0:
        return "FIST"
    if extended_count == 5:
        return "OPEN"
    if dist_thumb_index < 30:
        # Se middle, ring e pinky também estiverem esticados, "OK" clássico
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

# Sequência de gestos para a mão ESQUERDA
gesture_sequence_left = []

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Por padrão, nenhum filtro é aplicado
    current_filter = "NONE"
    guardar_imagem = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Identificar se é mão esquerda ou direita
            handed_label = results.multi_handedness[hand_idx].classification[0].label
            if handed_label == "Left":
                mao_str = "Mao Esquerda"
            else:
                mao_str = "Mao Direita"

            # Converter landmarks para coordenadas (x, y) em pixels
            lm_list = []
            for lm in hand_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)
                lm_list.append((px, py))

            # Desenhar a mão
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar o gesto
            gesto = detect_gesture(lm_list)

            # ===============================
            #   APLICAR FILTRO (apenas E)
            # ===============================
            # Se for PINCH e a mão for ESQUERDA, aplicamos GRAY
            if gesto == "PINCH" and mao_str == "Mao Esquerda":
                current_filter = "GRAY"

            # ===============================
            #   DETECTAR SEQUÊNCIA (apenas E)
            # ===============================
            # Se a mão for direita e o gesto for "FIST" ou "OPEN"
            if mao_str == "Mao Direita" and gesto in ["FIST", "OPEN"]:
                # Só adiciona se for diferente do último registro,
                # evitando repetições consecutivas do mesmo gesto
                if len(gesture_sequence_left) == 0 or gesto != gesture_sequence_left[-1]:
                    gesture_sequence_left.append(gesto)

                # Mantém só os últimos 3
                if len(gesture_sequence_left) > 3:
                    gesture_sequence_left.pop(0)

                # Verifica se a lista formou a sequência [FIST, OPEN, FIST]
                if gesture_sequence_left == ["FIST", "OPEN", "FIST"]:
                    guardar_imagem = True
                    
                    # Limpa a lista para permitir nova captura se quiser
                    gesture_sequence_left.clear()

            # Mostrar na tela qual gesto e qual mão
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

    # ===================================
    # APLICA O FILTRO (SE DEFINIDO)
    # ===================================
    if current_filter == "GRAY":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif current_filter == "BGR":
        # Aqui você poderia implementar outro tipo de filtro,
        # mas por ora deixamos como BGR normal.
        pass

   # ===================================
    # Guardar (SE DEFINIDO)
    # ===================================
    if guardar_imagem:
        guardar_imagem = False
        # guardar a imagem (já filtrada se o filtro estiver ligado)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(filename, frame)
        print(f"Imagem guardada: {filename}")

    # Exibir o resultado
    cv2.imshow("Detecao de Maos e Gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
