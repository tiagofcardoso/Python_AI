import cv2
import numpy as np
import cv2.dnn as dnn

# Carregar o modelo YOLO
net = cv2.dnn.readNet("./config_files/yolov3.weights", "./config_files/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Carregar os nomes das classes
with open("./config_files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definir cores para cada classe
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Capturar vídeo da câmera do laptop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    # Preparar a imagem para o modelo YOLO
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analisar as saídas do modelo
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression para eliminar caixas sobrepostas
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Desenhar as caixas delimitadoras e rótulos
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibir o frame com as detecções
    cv2.imshow('Frame', frame)

    # Sair do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar janelas
cap.release()
cv2.destroyAllWindows()
