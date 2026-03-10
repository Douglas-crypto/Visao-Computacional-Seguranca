import cv2
import time
import numpy as np
from datetime import datetime

def salvar_log(tipo):
    data_hora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open("relatorio_contagem.txt", "a") as f:
        f.write(f"[{data_hora}] - Evento: {tipo}\n")

def contador_ultra_sensivel():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    # --- NOVIDADE: SUBTRATOR DE FUNDO INTELIGENTE ---
    # Isso ajuda a ignorar mudanças de luz e focar apenas no que se move
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    entradas = 0
    saidas = 0
    linha_y = 300 
    offset = 45   
    na_zona_de_espera = False 
    veio_de_cima = False

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Aplica o subtrator de fundo (se adapta a luz sozinho)
        fgmask = fgbg.apply(frame)
        
        # Limpa o ruído (pontinhos brancos)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroide_y = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000 and area > max_area: # MAIS SENSÍVEL (2000 em vez de 5000)
                max_area = area
                (x, y, w, h) = cv2.boundingRect(contour)
                centroide_y = y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (x + w // 2, centroide_y), 5, (0, 0, 255), -1)

        # --- LÓGICA DE CONTAGEM (MESMA DE ANTES) ---
        if centroide_y is not None:
            if centroide_y > (linha_y - offset) and centroide_y < (linha_y + offset):
                if not na_zona_de_espera:
                    na_zona_de_espera = True
                    veio_de_cima = True if centroide_y < linha_y else False
            elif na_zona_de_espera and veio_de_cima and centroide_y > (linha_y + offset):
                entradas += 1
                salvar_log("ENTRADA")
                na_zona_de_espera = False
            elif na_zona_de_espera and not veio_de_cima and centroide_y < (linha_y - offset):
                saidas += 1
                salvar_log("SAIDA")
                na_zona_de_espera = False
        else:
            na_zona_de_espera = False

        # --- INTERFACE ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, linha_y - offset), (frame.shape[1], linha_y + offset), (255, 100, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, f"ENTRADAS: {entradas}", (20, 50), 1, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"SAIDAS:   {saidas}", (20, 100), 1, 2, (0, 0, 255), 2)
        
        cv2.imshow('Vision IA - Ultra Sensivel', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): entradas = saidas = 0
        if key == ord('w'): linha_y -= 15
        if key == ord('s'): linha_y += 15

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    contador_ultra_sensivel()