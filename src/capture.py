import cv2
import time
import numpy as np
from datetime import datetime

def salvar_log(tipo):
    data_hora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open("relatorio_contagem.txt", "a") as f:
        f.write(f"[{data_hora}] - Evento: {tipo}\n")

def contador_profissional_v2():
    print("=== Vision IA 2026: Contador Blindado v2 ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    time.sleep(2)
    
    first_frame = None
    entradas = 0
    saidas = 0
    
    # --- CONFIGURAÇÃO DA ZONA MORTA ---
    linha_y = 300 
    offset = 40   # Faixa de segurança (aumentada para 40 para ser mais visível)
    
    # Variáveis de controle de estado
    na_zona_de_espera = False 
    veio_de_cima = False

    cv2.namedWindow('Vision IA - Profissional', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break

            # --- PROCESSAMENTO ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if first_frame is None:
                first_frame = gray
                continue

            frame_diff = cv2.absdiff(first_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroide_y = None
            max_area = 0
            melhor_contorno = None

            # FOCO NO MAIOR OBJETO (Para evitar confusão com o fundo)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000 and area > max_area:
                    max_area = area
                    melhor_contorno = contour

            if melhor_contorno is not None:
                (x, y, w, h) = cv2.boundingRect(melhor_contorno)
                centroide_y = y + h // 2
                
                # Desenho do objeto principal
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (x + w // 2, centroide_y), 5, (0, 0, 255), -1)

            # --- LÓGICA DE ZONA MORTA ---
            if centroide_y is not None:
                # 1. Entrou na Zona Morta
                if centroide_y > (linha_y - offset) and centroide_y < (linha_y + offset):
                    if not na_zona_de_espera:
                        na_zona_de_espera = True
                        # Marca se ele entrou por cima ou por baixo da zona azul
                        veio_de_cima = True if centroide_y < linha_y else False
                
                # 2. Saiu por baixo (Confirma Entrada)
                elif na_zona_de_espera and veio_de_cima and centroide_y > (linha_y + offset):
                    entradas += 1
                    salvar_log("ENTRADA")
                    na_zona_de_espera = False
                    print(f"📥 Entrada: {entradas}")

                # 3. Saiu por cima (Confirma Saída)
                elif na_zona_de_espera and not veio_de_cima and centroide_y < (linha_y - offset):
                    saidas += 1
                    salvar_log("SAIDA")
                    na_zona_de_espera = False
                    print(f"📤 Saida: {saidas}")
            else:
                # Se o objeto sumir antes de atravessar a zona, cancela a espera
                na_zona_de_espera = False 

            # --- INTERFACE VIP (EFEITO VIDRO) ---
            # 1. Zona Morta Azulada
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, linha_y - offset), (frame.shape[1], linha_y + offset), (255, 150, 0), -1)
            
            # 2. Painel de Contagem Superior (Transparente)
            cv2.rectangle(overlay, (0, 0), (280, 115), (30, 30, 30), -1)
            
            # Aplica a transparência em tudo que está no 'overlay'
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) 
            
            # Textos
            cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (255, 255, 255), 1)
            cv2.putText(frame, f"ENTRADAS: {entradas}", (20, 45), 1, 1.8, (0, 255, 0), 2)
            cv2.putText(frame, f"SAIDAS:   {saidas}", (20, 95), 1, 1.8, (0, 0, 255), 2)
            
            cv2.putText(frame, "W/S: Mover Linha | R: Reset | Q: Sair", (10, frame.shape[0]-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow('Vision IA - Profissional', frame)

            # --- CONTROLES ---
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                entradas = saidas = 0
                first_frame = None
                print("Resetando...")
            elif key == ord('w'): linha_y -= 15
            elif key == ord('s'): linha_y += 15

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    contador_profissional_v2()