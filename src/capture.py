import cv2
import time

def testar_camera():
    print("=== Teste de Captura com Travas de Segurança ===")
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            print("Erro: Câmera não encontrada.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Exibe o frame
            cv2.imshow('Vision IA - Pressione Q para Sair', frame)

            # Trava de segurança 1: Espera 1ms e verifica a tecla
            # Trava de segurança 2: Se a janela for fechada no 'X', o script para
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('Vision IA - Pressione Q para Sair', cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Trava de segurança 3: Pequena pausa para não fritar a CPU
            time.sleep(0.01) 

    except KeyboardInterrupt:
        print("\n\nParada forçada pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"Erro inesperado: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Comando extra para garantir que as janelas sumam no Linux
        cv2.waitKey(1) 
        print("Sistema encerrado com segurança.")

if __name__ == "__main__":
    testar_camera()