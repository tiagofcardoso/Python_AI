import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Carregar o modelo de segmentação YOLO
model = YOLO("yolo11x-seg.pt")

# Captura de vídeo
cap =  cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)

# Obter as dimensões e fps (frames por segundo) do vídeo de entrada
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Criar um escritor de vídeo para guardar o resultado do processamento
out = cv2.VideoWriter("instance-segmentation-object-tracking.avi",
                      cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))


# Ciclo principal para ler e processar cada fotograma do vídeo
while True:
    ret, im0 = cap.read()
    if not ret:
        # Se não for possível ler mais fotogramas, sair do ciclo
        break

    # Inicializar o objeto 'annotator' para desenhar máscaras e anotações no fotograma
    annotator = Annotator(im0, line_width=5)

    # Executar o tracking (acompanhamento) de objetos no fotograma atual
    results = model.track(im0, persist=True)

    # Se existirem deteções com IDs e máscaras, desenhar as máscaras no fotograma
    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Para cada objeto detetado, desenhar a máscara, cor e etiqueta (ID)
        for mask, track_id in zip(masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color,
                               label=str(track_id),
                               txt_color=txt_color)
            
    # Escrever o fotograma anotado no vídeo de saída
    out.write(im0)

    # Mostrar o fotograma anotado numa janela
    cv2.imshow("instance-segmentation-object-tracking", im0)


    # Se o utilizador carregar na tecla 'q', sair do ciclo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libertar o escritor de vídeo e o objeto de captura de vídeo
out.release()
cap.release()
# Fechar todas as janelas abertas
cv2.destroyAllWindows()