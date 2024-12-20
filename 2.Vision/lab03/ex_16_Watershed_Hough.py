import cv2
import torch
import numpy as np

# Carrega o modelo YOLOv5 pré-treinado (ou um modelo treinado especificamente)
# ou um modelo customizado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.eval()


def detect_objects(frame, classes_to_detect):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = [img]
    results = model(img)
    objects = []
    for *xyxy, conf, cls in results.xyxy[0].tolist():
      if int(cls) in classes_to_detect:
        objects.append([xyxy, int(cls)])
    return objects


def detect_pedestrians(frame):
    # 0 é a classe "pessoa"
    return [obj[0] for obj in detect_objects(frame, [0])]


def detect_traffic_signs(frame):
    # Supondo que a classe 1 seja "placa de trânsito"
    return detect_objects(frame, [1])


def detect_traffic_lights(frame):
    return detect_objects(frame, [2])  # Supondo que a classe 2 seja "semáforo"


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    roi_vertices = [(0, height), (width/2, height/2),
                    (width, height)]  # regiao de interesse
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, np.array([roi_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(canny, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180,
                            100, minLineLength=40, maxLineGap=5)

    lane_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_lines.append(line[0])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

    return frame, lane_lines

def main():
    video_capture = cv2.VideoCapture(2)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        pedestrians = detect_pedestrians(frame)
        frame, _ = detect_lanes(frame.copy())
        traffic_signs = detect_traffic_signs(frame)
        traffic_lights = detect_traffic_lights(frame)

        for (x1, y1, x2, y2) in pedestrians:
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)
            if x1 < frame.shape[1] / 2 < x2:
                cv2.putText(frame, "ALERTA! PEDESTRE!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for xyxy, cls in traffic_signs:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, "Placa", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        for xyxy, cls in traffic_lights:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, "Semaforo", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow('Detecção Integrada', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
