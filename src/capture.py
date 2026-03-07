import cv2
import time
import numpy as np

def detectar_movimento():
    print("=== Vision IA 2026: Detector de Movimento ===")
    
    # 1. Abrir a câmera APENAS UMA VEZ fora do loop
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    # 2. Aguardar a câmera estabilizar
    print("Aguardando inicialização do hardware...")
    time.sleep(2)

    first_frame = None
    min_area = 1000 

    # Criar a janela UMA VEZ antes do loop para evitar múltiplas janelas
    cv2.namedWindow('Vision IA - Detector', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # --- PROCESSAMENTO ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if first_frame is None:
                first_frame = gray
                print("Fundo de referência definido.")
                continue

            # Cálculo de diferença matemática
            frame_diff = cv2.absdiff(first_frame, gray)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # --- DETECÇÃO ---
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            movimento = False
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                movimento = True

            # --- INTERFACE ---
            status = "MOVIMENTO!" if movimento else "ESTATICO"
            cor = (0, 0, 255) if movimento else (0, 255, 0)
            cv2.putText(frame, f"SISTEMA: {status}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            
            # 3. Mostrar o frame na janela já criada
            cv2.imshow('Vision IA - Detector', frame)

            # --- CONTROLES ---
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                first_frame = None
                print("Resetando fundo...")

    except Exception as e:
        print(f"Erro: {e}")
    finally:
        # 4. Fechamento seguro
        cap.release()
        cv2.destroyAllWindows()
        # Garante que o Linux feche a janela gráfica
        for _ in range(10):
            cv2.waitKey(1)
        print("✓ Recursos liberados.")

if __name__ == "__main__":
    detectar_movimento()