import cv2
import numpy as np
import math
from datetime import datetime
import os
import time

# --- CONFIGURAÇÃO DE PASTAS ---
if not os.path.exists('logs'): os.makedirs('logs')
if not os.path.exists('capturas'): os.makedirs('capturas')

# Variável global para controle de cadência da voz
ultimo_alarme_txt = 0

class Rastreador:
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
        
        nova_centros_objetos = {obj_id[4]: self.centros_objetos[obj_id[4]] for obj_id in objetos_id}
        self.centros_objetos = nova_centros_objetos.copy()
        return objetos_id

def disparar_alarme(tipo="evento"):
    """
    Emite alerta sonoro. 
    'perigo' usa síntese de voz curta e direta: "Alerta. Limite."
    """
    global ultimo_alarme_txt
    agora = time.time()
    
    if tipo == "perigo":
        # Intervalo de 4 segundos para a voz não encavalar
        if agora - ultimo_alarme_txt > 3:
            # -r -30: velocidade equilibrada | -l pt: força o idioma português
            os.system('spd-say "Capacidade maxima." -r -30 -p 5 -t male1 -l pt &')
            ultimo_alarme_txt = agora
        
        # Som de erro do sistema (feedback imediato)
        os.system('paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga &')
    else:
        # Som de clique discreto para entradas/saídas normais
        os.system('paplay /usr/share/sounds/freedesktop/stereo/message-new-instant.oga &')

def salvar_dados(tipo, ocupacao, frame, alerta=False):
    agora = datetime.now()
    data = agora.strftime('%d/%m/%Y')
    hora = agora.strftime('%H:%M:%S')
    timestamp_foto = agora.strftime('%Y%m%d_%H%M%S_%f')
    
    status_alerta = "ALERTA_LOTADO" if alerta else "NORMAL"
    caminho_csv = "logs/dados_fluxo.csv"
    
    with open(caminho_csv, "a") as f:
        if os.stat(caminho_csv).st_size == 0:
            f.write("Data;Hora;Evento;Ocupacao;Status\n")
        f.write(f"{data};{hora};{tipo};{ocupacao};{status_alerta}\n")
    
    cv2.imwrite(f"capturas/{tipo}_{timestamp_foto}.jpg", frame)

def sistema_seguranca_ativa():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    rastreador = Rastreador()

    ret, frame_teste = cap.read()
    if not ret: return
    
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    linha_y = altura // 2

    entradas, saidas = 0, 0
    limite_maximo = 5 # Defina aqui sua capacidade máxima
    offset = 45
    ids_contados = set() 

    print(f"Sistema Ativo. Limite: {limite_maximo}. Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_limpo = frame.copy()

        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        deteccoes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 3500: 
                deteccoes.append(cv2.boundingRect(cnt))

        info_objetos = rastreador.atualizar(deteccoes)
        ocupacao_viva = max(0, entradas - saidas)

        for obj in info_objetos:
            x, y, w, h, id = obj
            cy = y + h // 2
            
            if id not in ids_contados:
                # LÓGICA DE ENTRADA
                if cy > (linha_y + offset):
                    entradas += 1
                    ids_contados.add(id)
                    lotado = (entradas - saidas) >= limite_maximo
                    salvar_dados("ENTRADA", entradas - saidas, frame_limpo, lotado)
                    disparar_alarme("perigo" if lotado else "evento")
                
                # LÓGICA DE SAÍDA
                elif cy < (linha_y - offset):
                    if (entradas - saidas) > 0:
                        saidas += 1
                        ids_contados.add(id)
                        salvar_dados("SAIDA", entradas - saidas, frame_limpo, False)
                        disparar_alarme("evento")
                    else:
                        ids_contados.add(id)

        # --- INTERFACE DE ALERTA E DASHBOARD ---
        ocupacao_viva = max(0, entradas - saidas)
        esta_lotado = ocupacao_viva >= limite_maximo
        cor_status = (0, 0, 255) if esta_lotado else (0, 255, 0)

        # Feedback visual de pânico/alerta (Borda piscante)
        if esta_lotado and int(time.time()) % 2 == 0:
            cv2.rectangle(frame, (0,0), (largura, altura), (0, 0, 255), 15)
            cv2.putText(frame, "CAPACIDADE MAXIMA!", (largura//2 - 180, altura - 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        # Dashboard com Alpha Blending (Transparência)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Linha e Textos
        cv2.line(frame, (0, linha_y), (largura, linha_y), (255, 255, 255), 2)
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"OCUPACAO: {ocupacao_viva}", (20, 45), fonte, 1.2, cor_status, 2)
        cv2.putText(frame, f"LIMITE: {limite_maximo} | IN:{entradas} OUT:{saidas}", (20, 90), fonte, 0.7, (255,255,255), 1)

        cv2.imshow('Vision IA - Passo 9 (Seguranca Ativa)', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord('r'): # Reset manual
            entradas = saidas = 0
            ids_contados.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sistema_seguranca_ativa()