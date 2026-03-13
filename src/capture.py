import cv2
import numpy as np
import math
from datetime import datetime
import os

# --- CONFIGURAÇÃO DE PASTAS ---
if not os.path.exists('logs'): os.makedirs('logs')
if not os.path.exists('capturas'): os.makedirs('capturas')

class Rastreador:
    """
    Classe responsável por atribuir IDs únicos aos objetos e 
    rastreá-los entre os frames usando Distância Euclidiana.
    """
    def __init__(self):
        self.centros_objetos = {} 
        self.id_count = 0

    def atualizar(self, objetos_detectados):
        objetos_id = []
        for rect in objetos_detectados:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            objeto_ja_rastreado = False
            for id, pt in self.centros_objetos.items():
                distancia = math.hypot(cx - pt[0], cy - pt[1])

                if distancia < 60: 
                    self.centros_objetos[id] = (cx, cy)
                    objetos_id.append([x, y, w, h, id])
                    objeto_ja_rastreado = True
                    break

            if not objeto_ja_rastreado:
                self.centros_objetos[self.id_count] = (cx, cy)
                objetos_id.append([x, y, w, h, self.id_count])
                self.id_count += 1

        nova_centros_objetos = {}
        for obj_id in objetos_id:
            _, _, _, _, id = obj_id
            nova_centros_objetos[id] = self.centros_objetos[id]
        self.centros_objetos = nova_centros_objetos.copy()
        
        return objetos_id

def salvar_dados(tipo, ocupacao, frame):
    """Registra o evento no CSV e tira uma foto de prova."""
    agora = datetime.now()
    data = agora.strftime('%d/%m/%Y')
    hora = agora.strftime('%H:%M:%S')
    timestamp_foto = agora.strftime('%Y%m%d_%H%M%S_%f')
    
    caminho_csv = "logs/dados_fluxo.csv"
    file_exists = os.path.isfile(caminho_csv)
    
    with open(caminho_csv, "a") as f:
        if not file_exists or os.stat(caminho_csv).st_size == 0:
            f.write("Data;Hora;Evento;Ocupacao\n")
        f.write(f"{data};{hora};{tipo};{ocupacao}\n")
    
    cv2.imwrite(f"capturas/{tipo}_{timestamp_foto}.jpg", frame)

def sistema_tracking_avancado():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    rastreador = Rastreador()

    # --- AJUSTE RESPONSIVO DA LINHA CENTRAL ---
    ret, frame_teste = cap.read()
    if ret:
        altura_camera = frame_teste.shape[0]
        linha_y = altura_camera // 2
    else:
        linha_y = 240 # Fallback padrão

    entradas = 0
    saidas = 0
    limite_capacidade = 5
    offset = 45 # Margem de erro para a contagem na linha
    ids_contados = set() 

    print(f"Sistema Iniciado. Linha central em: {linha_y}px. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_limpo = frame.copy()

        # --- PROCESSAMENTO ---
        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        deteccoes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 3500: 
                x, y, w, h = cv2.boundingRect(cnt)
                deteccoes.append([x, y, w, h])

        # --- RASTREAMENTO ---
        info_objetos = rastreador.atualizar(deteccoes)

        for obj in info_objetos:
            x, y, w, h, id = obj
            centroide_y = y + h // 2
            
            cv2.putText(frame, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- LÓGICA DE CONTAGEM ---
            if id not in ids_contados:
                # ENTRADA (Cima para Baixo)
                if centroide_y > (linha_y + offset):
                    entradas += 1
                    ids_contados.add(id)
                    ocupacao_atual = max(0, entradas - saidas)
                    salvar_dados("ENTRADA", ocupacao_atual, frame_limpo)
                
                # SAÍDA (Baixo para Cima)
                elif centroide_y < (linha_y - offset):
                    if (entradas - saidas) > 0: # Trava de segurança
                        saidas += 1
                        ids_contados.add(id)
                        ocupacao_atual = max(0, entradas - saidas)
                        salvar_dados("SAIDA", ocupacao_atual, frame_limpo)
                    else:
                        # Ignora saída se contador for zero, mas marca ID como visto
                        ids_contados.add(id)

        # --- INTERFACE VISUAL (DASHBOARD TRANSPARENTE) ---
        ocupacao_viva = max(0, entradas - saidas)
        cor_status = (0, 255, 0) if ocupacao_viva < limite_capacidade else (0, 0, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) # Transparência suave

        # Linha central branca
        cv2.line(frame, (0, linha_y), (frame.shape[1], linha_y), (255, 255, 255), 2)
        
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"OCUPACAO: {ocupacao_viva}", (20, 40), fonte, 1.0, cor_status, 2)
        cv2.putText(frame, f"IN: {entradas} | OUT: {saidas}", (20, 80), fonte, 0.6, (255, 255, 255), 2)

        cv2.imshow('Vision IA - Passo 8 (Tracking)', frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): 
            entradas = saidas = 0
            ids_contados.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sistema_tracking_avancado()