import cv2
import time
import numpy as np
from datetime import datetime

def salvar_log(tipo, ocupacao):
    data_hora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open("relatorio_contagem.txt", "a") as f:
        f.write(f"[{data_hora}] - Evento: {tipo} | Ocupacao Atual: {ocupacao}\n")

def sistema_bi_ia_harmonizado():
    print("=== Vision IA 2026: Dashboard Harmonizado (Passo 6) ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    # Algoritmo MOG2 para adaptacao de luz (do Passo 5)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    entradas = 0
    saidas = 0
    limite_capacidade = 5
    
    linha_y = 300 
    offset = 45   
    na_zona_de_espera = False 
    veio_de_cima = False

    cv2.namedWindow('Vision IA - Dashboard', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break

            # --- PROCESSAMENTO INTELIGENTE ---
            fgmask = fgbg.apply(frame)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroide_y = None
            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2500 and area > max_area: 
                    max_area = area
                    (x, y, w, h) = cv2.boundingRect(contour)
                    centroide_y = y + h // 2
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (x + w // 2, centroide_y), 5, (0, 0, 255), -1)

            # --- LOGICA DE CONTAGEM ---
            ocupacao = max(0, entradas - saidas)
            if centroide_y is not None:
                if (linha_y - offset) < centroide_y < (linha_y + offset):
                    if not na_zona_de_espera:
                        na_zona_de_espera = True
                        veio_de_cima = True if centroide_y < linha_y else False
                elif na_zona_de_espera and veio_de_cima and centroide_y > (linha_y + offset):
                    entradas += 1
                    ocupacao = max(0, entradas - saidas)
                    salvar_log("ENTRADA", ocupacao)
                    na_zona_de_espera = False
                elif na_zona_de_espera and not veio_de_cima and centroide_y < (linha_y - offset):
                    saidas += 1
                    ocupacao = max(0, entradas - saidas)
                    salvar_log("SAIDA", ocupacao)
                    na_zona_de_espera = False
            else:
                na_zona_de_espera = False

            # --- DEFINICAO DE STATUS E CORES ---
            cor_status = (0, 255, 0) # Verde
            status_texto = "NORMAL"
            if ocupacao >= limite_capacidade:
                cor_status = (0, 0, 255) # Vermelho
                status_texto = "CAPACIDADE MAXIMA!"
            elif ocupacao >= limite_capacidade - 2:
                cor_status = (0, 255, 255) # Amarelo
                status_texto = "ALERTA: QUASE CHEIO"

            # --- INTERFACE HARMONIZADA (UX) ---
            overlay = frame.copy()
            # Painel lateral mais elegante
            cv2.rectangle(overlay, (0, 0), (300, 160), (30, 30, 30), -1)
            # Faixa da Zona Morta
            cv2.rectangle(overlay, (0, linha_y - offset), (frame.shape[1], linha_y + offset), (255, 120, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Textos com fontes e tamanhos equilibrados
            # Fonte: FONT_HERSHEY_SIMPLEX para um look moderno
            cv2.putText(frame, f"IN: {entradas} | OUT: {saidas}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"OCUPACAO: {ocupacao}", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor_status, 2)
            
            cv2.putText(frame, f"STATUS: {status_texto}", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
            
            # Linha guia central
            cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (255, 255, 255), 1)
            
            # Atalhos no rodape
            cv2.putText(frame, "W/S: Mover Linha | R: Reset | Q: Sair", (10, frame.shape[0]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow('Vision IA - Dashboard', frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord('r'): entradas = saidas = 0
            elif key == ord('w'): linha_y -= 15
            elif key == ord('s'): linha_y += 15

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sistema_bi_ia_harmonizado()