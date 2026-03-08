import cv2
import time
import numpy as np

def contador_objetos():
    print("=== Vision IA 2026: Contador de Objetos (Passo 3) ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao acessar a câmera.")
        return

    time.sleep(2) # Estabilização
    
    first_frame = None
    total_counts = 0
    linha_y = 300 # Altura da linha (ajustaremos no primeiro frame)
    posicao_anterior = None # Guarda o Y do objeto no frame passado

    # Criar a janela ANTES para estabilidade no Linux
    cv2.namedWindow('Vision IA - Contador', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Ajusta a linha para o meio da tela dinamicamente
            if linha_y == 300:
                linha_y = frame.shape[0] // 2

            # --- PROCESSAMENTO ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if first_frame is None:
                first_frame = gray
                continue

            frame_diff = cv2.absdiff(first_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # --- DETECÇÃO ---
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            movimento_atual_y = None

            for contour in contours:
                if cv2.contourArea(contour) < 2000: # Ignora ruídos pequenos
                    continue
                
                (x, y, w, h) = cv2.boundingRect(contour)
                Cx = x + w // 2
                Cy = y + h // 2
                
                # Desenha o objeto e o centroide
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (Cx, Cy), 5, (0, 0, 255), -1)
                
                # Pegamos o Cy do maior objeto para monitorar o cruzamento
                movimento_atual_y = Cy

            # --- LÓGICA DE CONTAGEM (O PULO DO GATO) ---
            if movimento_atual_y is not None and posicao_anterior is not None:
                # Se no frame passado estava ACIMA e agora está ABAIXO da linha
                if posicao_anterior < linha_y and movimento_atual_y >= linha_y:
                    total_counts += 1
                    print(f"Objeto Cruzou! Total: {total_counts}")

            posicao_anterior = movimento_atual_y

            # --- INTERFACE ---
            # Linha Amarela
            cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (0, 255, 255), 2)
            # Contador Gigante
            cv2.putText(frame, f"CONTADOR: {total_counts}", (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 0), 3)
            
            cv2.imshow('Vision IA - Contador', frame)

            # --- CONTROLES ---
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                first_frame = None
                total_counts = 0
                print("Resetando...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    contador_objetos()