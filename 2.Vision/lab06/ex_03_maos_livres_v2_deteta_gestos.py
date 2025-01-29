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

    # Mapeamento dos índices das landmarks conforme o Mediapipe:
    # 0: wrist
    # Thumb: 1 (CMC), 2 (MCP), 3 (IP), 4 (TIP)
    # Index: 5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
    # Middle: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
    # Ring: 13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
    # Pinky: 17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)

    # Para cada dedo, vamos pegar a tripla (MCP, PIP, TIP) ou (MCP, IP, TIP) no caso do polegar
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

    # Extrair (x, y) da mão
    mcp = hand_landmarks[mcp_idx]
    pip_ = hand_landmarks[pip_idx]
    tip = hand_landmarks[tip_idx]

    # Calcular o ângulo no PIP (ou IP) entre MCP e TIP
    ang = angle_3points(mcp, pip_, tip)

    # Definir um limiar para dizer se está "esticado" (podemos ajustar ~160-170)
    if ang > 160:
        return True
    else:
        return False

def detect_gesture(lm_list):
    """
    Detecta gesto a partir do estado (esticado ou não) de cada dedo.
    Gera um "mapa" dos dedos esticados para comparar com gestos pré-definidos.
    """

    # Verificação dedo a dedo
    thumb_extended  = finger_is_extended(lm_list, "thumb")
    index_extended  = finger_is_extended(lm_list, "index")
    middle_extended = finger_is_extended(lm_list, "middle")
    ring_extended   = finger_is_extended(lm_list, "ring")
    pinky_extended  = finger_is_extended(lm_list, "pinky")

    # Contagem de dedos esticados
    extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    extended_count = sum(extended_fingers)

    # Para gestos que dependem de distância específica entre polegar e indicador
    # (por exemplo, "OK", "PINCH"), vamos calcular:
    thumb_tip = lm_list[4]
    index_tip = lm_list[8]
    dist_thumb_index = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])

    # ---- Exemplos de gestos ----
    # 1) FIST: nenhum dedo esticado (extended_count = 0)
    if extended_count == 0:
        return "FIST"

    # 2) OPEN: todos os dedos esticados (extended_count = 5)
    if extended_count == 5:
        return "OPEN"

    # 3) OK: polegar e indicador juntos + (3 dedos esticados ou não)
    #    Geralmente, o "OK" clássico tem polegar e indicador juntos e os outros dedos esticados.
    #    Mas vamos deixar mais flexível se dist < 30 ou outro limiar.
    if dist_thumb_index < 30:
        # Se middle, ring e pinky também estiverem esticados, OK "clássico"
        if middle_extended and ring_extended and pinky_extended:
            return "OK"
        else:
            return "PINCH"

    # 4) POINT: somente o indicador esticado
    #    (outra heurística: polegar semi-aberto, mas para simplificar deixamos só index=True)
    if index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        return "POINT"

    # 5) THUMBS_UP: somente o polegar esticado
    #    Observação: "thumbs up" real seria o polegar para cima e os outros fechados
    #    mas vamos simplificar com a heurística do dedo estendido.
    if thumb_extended and (not index_extended) and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        return "THUMBS_UP"

    # 6) PEACE (ou "V"): indicador e médio esticados, anelar e mindinho dobrados
    if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
        return "PEACE"

    # 7) ROCK: indicador e mindinho esticados, médio e anelar dobrados
    if index_extended and (not middle_extended) and (not ring_extended) and pinky_extended:
        return "ROCK"

    # Podemos criar muitos outros mapas de bits para diferentes gestos...
    # O else se não encaixar em nenhum "padrão" simples
    return "DESCONHECIDO"

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

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

            # Detetar o gesto com a nova função de ângulos
            gesto = detect_gesture(lm_list)

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

    cv2.imshow("Detecao de Maos e Gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
