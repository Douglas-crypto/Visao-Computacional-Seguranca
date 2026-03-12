import cv2
import time
import numpy as np
from datetime import datetime
import os

# --- CONFIGURAÇÃO DE PASTAS ---
if not os.path.exists('logs'): os.makedirs('logs')
if not os.path.exists('capturas'): os.makedirs('capturas')

def salvar_dados(tipo, ocupacao, frame):
    agora = datetime.now()
    data_hora = agora.strftime('%d/%m/%Y %H:%M:%S')
    timestamp_arquivo = agora.strftime('%Y%m%d_%H%M%S_%f')
    
    # 1. Salvar no TXT (Leitura Humana)
    with open("logs/relatorio_geral.txt", "a") as f:
        f.write(f"[{data_hora}] - {tipo} | Ocupação: {ocupacao}\n")
    
    # 2. Salvar no CSV (Excel)
    file_exists = os.path.isfile('logs/dados_fluxo.csv')
    with open("logs/dados_fluxo.csv", "a") as f:
        if not file_exists:
            f.write("Data;Hora;Evento;Ocupacao\n") # Cabeçalho
        f.write(f"{agora.strftime('%d/%m/%Y')};{agora.strftime('%H:%M:%S')};{tipo};{ocupacao}\n")

    # 3. Salvar Foto da Evidência
    nome_foto = f"capturas/{tipo}_{timestamp_arquivo}.jpg"
    cv2.imwrite(nome_foto, frame)
    return True

def sistema_auditoria_ia():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    entradas = 0
    saidas = 0
    limite_capacidade = 5
    linha_y = 300 
    offset = 45   
    na_zona_de_espera = False 
    veio_de_cima = False
    
    msg_foto_timer = 0 # Timer para mostrar aviso na tela

    while True:
        ret, frame = cap.read()
        if not ret: break
        copy_evidencia = frame.copy() # Cópia limpa para a foto de auditoria

        # Processamento
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

        # Lógica de Contagem e Disparo de Auditoria
        ocupacao = max(0, entradas - saidas)
        if centroide_y is not None:
            if (linha_y - offset) < centroide_y < (linha_y + offset):
                if not na_zona_de_espera:
                    na_zona_de_espera = True
                    veio_de_cima = True if centroide_y < linha_y else False
            
            elif na_zona_de_espera and veio_de_cima and centroide_y > (linha_y + offset):
                entradas += 1
                ocupacao = max(0, entradas - saidas)
                salvar_dados("ENTRADA", ocupacao, copy_evidencia)
                na_zona_de_espera = False
                msg_foto_timer = 20 # Mostra aviso por 20 frames

            elif na_zona_de_espera and not veio_de_cima and centroide_y < (linha_y - offset):
                saidas += 1
                ocupacao = max(0, entradas - saidas)
                salvar_dados("SAIDA", ocupacao, copy_evidencia)
                na_zona_de_espera = False
                msg_foto_timer = 20
        else:
            na_zona_de_espera = False

        # --- INTERFACE ---
        cor_status = (0, 255, 0)
        status_texto = "NORMAL"
        if ocupacao >= limite_capacidade:
            cor_status = (0, 0, 255); status_texto = "CAPACIDADE MAXIMA!"
        elif ocupacao >= limite_capacidade - 2:
            cor_status = (0, 255, 255); status_texto = "ALERTA: QUASE CHEIO"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 165), (30, 30, 30), -1)
        cv2.rectangle(overlay, (0, linha_y - offset), (frame.shape[1], linha_y + offset), (255, 120, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, f"IN: {entradas} | OUT: {saidas}", (20, 40), 1, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"OCUPACAO: {ocupacao}", (20, 85), 1, 1.0, cor_status, 2)
        cv2.putText(frame, f"STATUS: {status_texto}", (20, 130), 1, 0.6, cor_status, 2)
        
        # Aviso de Foto Salva
        if msg_foto_timer > 0:
            cv2.putText(frame, "EVIDENCIA SALVA!", (frame.shape[1]-200, 40), 1, 1.0, (255, 255, 0), 2)
            msg_foto_timer -= 1

        cv2.imshow('Vision IA - Auditoria', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): entradas = saidas = 0
        elif key == ord('w'): linha_y -= 15
        elif key == ord('s'): linha_y += 15

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sistema_auditoria_ia()