import cv2
import time
import numpy as np
from datetime import datetime

def salvar_log(tipo):
    """Função para salvar o evento em um arquivo de texto"""
    data_hora = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open("relatorio_contagem.txt", "a") as f:
        f.write(f"[{data_hora}] - Evento: {tipo}\n")

def contador_profissional():
    print("=== Vision IA 2026: Monitoramento Bidirecional ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    time.sleep(2)
    
    first_frame = None
    entradas = 0
    saidas = 0
    linha_y = 300
    posicao_anterior = None

    cv2.namedWindow('Vision IA - Monitoramento', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break

            if linha_y == 300: linha_y = frame.shape[0] // 2

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
            
            movimento_atual_y = None

            for contour in contours:
                if cv2.contourArea(contour) < 3000: continue # Área um pouco maior para evitar erros
                (x, y, w, h) = cv2.boundingRect(contour)
                Cx = x + w // 2
                Cy = y + h // 2
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (Cx, Cy), 5, (0, 0, 255), -1)
                movimento_atual_y = Cy

            # --- LÓGICA BIDIRECIONAL ---
            if movimento_atual_y is not None and posicao_anterior is not None:
                # Caso 1: Cruzou de Cima para Baixo (ENTRADA)
                if posicao_anterior < linha_y and movimento_atual_y >= linha_y:
                    entradas += 1
                    salvar_log("ENTRADA")
                    print(f"📥 Entrada detectada! Total: {entradas}")

                # Caso 2: Cruzou de Baixo para Cima (SAÍDA)
                elif posicao_anterior > linha_y and movimento_atual_y <= linha_y:
                    saidas += 1
                    salvar_log("SAIDA")
                    print(f"📤 Saída detectada! Total: {saidas}")

            posicao_anterior = movimento_atual_y

            # --- INTERFACE ---
            # Linha de Divisão
            cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (255, 255, 0), 2)
            
            # Painel de Contagem
            cv2.putText(frame, f"ENTRADAS: {entradas}", (20, 50), 2, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"SAIDAS: {saidas}", (20, 90), 2, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Pressione 'R' para Reset ou 'Q' para Sair", (10, frame.shape[0]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Vision IA - Monitoramento', frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                first_frame = None
                entradas = saidas = 0
                print("Sistema Resetado.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    contador_profissional()