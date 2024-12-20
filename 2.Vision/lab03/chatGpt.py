import cv2
import numpy as np
import os
from ultralytics import YOLO

def nothing(x):
    pass

def criar_trackbars():
    # Cria janela para controles HSV
    cv2.namedWindow('Controles')
    
    # Trackbars para HSV
    cv2.createTrackbar('H Min', 'Controles', 0, 180, nothing)
    cv2.createTrackbar('H Max', 'Controles', 180, 180, nothing)
    cv2.createTrackbar('S Min', 'Controles', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Controles', 30, 255, nothing)
    cv2.createTrackbar('V Min', 'Controles', 200, 255, nothing)
    cv2.createTrackbar('V Max', 'Controles', 255, 255, nothing)
    
    # Trackbars para Hough
    cv2.createTrackbar('Threshold', 'Controles', 30, 100, nothing)
    cv2.createTrackbar('Min Line Length', 'Controles', 50, 200, nothing)
    cv2.createTrackbar('Max Line Gap', 'Controles', 100, 200, nothing)

def detectar_faixas(frame, roi_points=None):
    # Obtém valores HSV das trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Controles')
    h_max = cv2.getTrackbarPos('H Max', 'Controles')
    s_min = cv2.getTrackbarPos('S Min', 'Controles')
    s_max = cv2.getTrackbarPos('S Max', 'Controles')
    v_min = cv2.getTrackbarPos('V Min', 'Controles')
    v_max = cv2.getTrackbarPos('V Max', 'Controles')
    
    # Obtém valores Hough das trackbars
    threshold = cv2.getTrackbarPos('Threshold', 'Controles')
    min_line_length = cv2.getTrackbarPos('Min Line Length', 'Controles')
    max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Controles')
    
    # Converte para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define intervalo HSV
    lower_white = np.array([h_min, s_min, v_min])
    upper_white = np.array([h_max, s_max, v_max])
    
    # Cria máscara HSV
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Aplica ROI se fornecido
    if roi_points is not None and len(roi_points) > 2:
        mask = np.zeros_like(white_mask)
        roi_points_array = np.array([roi_points], dtype=np.int32)
        cv2.fillPoly(mask, roi_points_array, 255)
        white_mask = cv2.bitwise_and(white_mask, mask)
    
    # Aplica blur
    blur = cv2.GaussianBlur(white_mask, (5, 5), 0)
    
    # Detecta bordas
    edges = cv2.Canny(blur, 50, 150)
    
    # Detecta linhas
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    # Desenha linhas
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return line_image, white_mask, edges

# Variáveis globais para ROI
roi_points = []
drawing = False

def mouse_callback(event, x, y, flags, param):
    global roi_points, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        roi_points = []

def processar_video():
    global roi_points
    
    # Carrega YOLO
    modelo = YOLO('yolov8n.pt')
    
    # Abre vídeo
    video_path = os.path.join(os.path.dirname(__file__), 'kart.mp4')
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print("Erro ao abrir o vídeo")
        return
    
    # Primeiro, cria todas as janelas necessárias
    cv2.namedWindow('Detecção', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Máscara', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Bordas', cv2.WINDOW_NORMAL)
    
    # Depois configura o callback do mouse
    cv2.setMouseCallback('Detecção', mouse_callback)
    
    # Por último, cria as trackbars
    criar_trackbars()
    
    print("Instruções:")
    print("- Clique com botão esquerdo para definir pontos ROI")
    print("- Clique com botão direito para limpar ROI")
    print("- Ajuste os controles HSV e Hough na janela 'Controles'")
    print("- Pressione 'q' para sair")
    print("- Pressione 's' para salvar os valores atuais")
    
    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reinicia o vídeo
                continue
            
            # Detecta faixas
            line_image, mask, edges = detectar_faixas(frame, roi_points)
            
            # Combina imagens
            resultado = cv2.addWeighted(frame, 1, line_image, 1, 0)
            
            # Desenha ROI atual
            if len(roi_points) > 1:
                pts = np.array(roi_points, np.int32)
                cv2.polylines(resultado, [pts], True, (255, 0, 0), 2)
            
            # Detecta karts
            deteccoes = modelo(frame)
            for r in deteccoes:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    nome_classe = modelo.names[cls]
                    
                    if conf > 0.3 and nome_classe in ['car', 'person']:
                        cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(resultado, f'{nome_classe} {conf:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 0, 255), 2)
            
            # Mostra resultados
            cv2.imshow('Detecção', resultado)
            cv2.imshow('Máscara', mask)
            cv2.imshow('Bordas', edges)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salva valores atuais
                print("\nValores HSV:")
                print(f"H: {cv2.getTrackbarPos('H Min', 'Controles')} - {cv2.getTrackbarPos('H Max', 'Controles')}")
                print(f"S: {cv2.getTrackbarPos('S Min', 'Controles')} - {cv2.getTrackbarPos('S Max', 'Controles')}")
                print(f"V: {cv2.getTrackbarPos('V Min', 'Controles')} - {cv2.getTrackbarPos('V Max', 'Controles')}")
                print("\nPontos ROI:")
                print(roi_points)
                print("\nParâmetros Hough:")
                print(f"Threshold: {cv2.getTrackbarPos('Threshold', 'Controles')}")
                print(f"Min Line Length: {cv2.getTrackbarPos('Min Line Length', 'Controles')}")
                print(f"Max Line Gap: {cv2.getTrackbarPos('Max Line Gap', 'Controles')}")
    
    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processar_video()