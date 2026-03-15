import cv2
import numpy as np
import math
from datetime import datetime
import os
import time
import threading
from flask import Flask, jsonify, render_template_string

# --- CONFIGURAÇÃO DE PASTAS ---
if not os.path.exists('logs'): os.makedirs('logs')
if not os.path.exists('capturas'): os.makedirs('capturas')

# --- VARIÁVEIS GLOBAIS (Ponte entre Câmera e Web) ---
dados_compartilhados = {
    "ocupacao": 0,
    "entradas": 0,
    "saidas": 0,
    "limite": 5,
    "status": "NORMAL"
}
ultimo_alarme_txt = 0

# --- CONFIGURAÇÃO DO SERVIDOR WEB (FLASK) ---
app = Flask(__name__)

@app.route('/')
def index():
    # Página Web que o seu celular vai abrir
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vision IA - Monitoramento</title>
        <meta http-equiv="refresh" content="1"> <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; background: #121212; color: white; margin: 0; padding: 20px; }
            .card { background: #1e1e1e; padding: 30px; border-radius: 20px; display: inline-block; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 3px solid #333; margin-top: 50px; }
            .valor { font-size: 80px; font-weight: bold; margin: 10px 0; }
            .status-ok { color: #00ff00; }
            .status-alerta { color: #ff3333; border-color: #ff3333; }
            .stats { font-size: 20px; color: #888; border-top: 1px solid #333; padding-top: 15px; }
        </style>
    </head>
    <body>
        <div class="card {{ 'status-alerta' if dados.status == 'ALERTA_LOTADO' else '' }}">
            <h2 style="margin:0;">LOTAÇÃO ATUAL</h2>
            <div class="valor {{ 'status-alerta' if dados.status == 'ALERTA_LOTADO' else 'status-ok' }}">
                {{ dados.ocupacao }}
            </div>
            <div class="stats">
                ENTRADAS: {{ dados.entradas }} | SAÍDAS: {{ dados.saidas }} <br>
                LIMITE: {{ dados.limite }}
            </div>
            {% if dados.status == 'ALERTA_LOTADO' %}
                <h3 style="color: #ff3333;"> CAPACIDADE MÁXIMA ATINGIDA!</h3>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html, dados=dados_compartilhados)

def rodar_servidor():
    # Roda na porta 5000. host='0.0.0.0' permite acesso via celular na mesma rede
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- CLASSES E FUNÇÕES DO SISTEMA DE VISÃO ---
class Rastreador:
    def __init__(self):
        self.centros_objetos = {} 
        self.id_count = 0

    def atualizar(self, objetos_detectados):
        objetos_id = []
        for rect in objetos_detectados:
            x, y, w, h = rect
            cx, cy = (x + x + w) // 2, (y + y + h) // 2
            objeto_ja_rastreado = False
            for id, pt in self.centros_objetos.items():
                if math.hypot(cx - pt[0], cy - pt[1]) < 60:
                    self.centros_objetos[id] = (cx, cy)
                    objetos_id.append([x, y, w, h, id])
                    objeto_ja_rastreado = True
                    break
            if not objeto_ja_rastreado:
                self.centros_objetos[self.id_count] = (cx, cy)
                objetos_id.append([x, y, w, h, self.id_count])
                self.id_count += 1
        self.centros_objetos = {obj[4]: self.centros_objetos[obj[4]] for obj in objetos_id}
        return objetos_id

def disparar_alarme(tipo="evento"):
    global ultimo_alarme_txt
    agora = time.time()
    if tipo == "perigo":
        if agora - ultimo_alarme_txt > 3:
            os.system('spd-say "Capacidade maxima" -r -30 -p 5 -t male1 -l pt &')
            ultimo_alarme_txt = agora
        os.system('paplay /usr/share/sounds/freedesktop/stereo/dialog-warning.oga &')
    else:
        os.system('paplay /usr/share/sounds/freedesktop/stereo/message-new-instant.oga &')

def salvar_dados(tipo, ocupacao, frame, alerta=False):
    agora = datetime.now()
    status = "ALERTA_LOTADO" if alerta else "NORMAL"
    caminho_csv = "logs/dados_fluxo.csv"
    with open(caminho_csv, "a") as f:
        if os.stat(caminho_csv).st_size == 0: f.write("Data;Hora;Evento;Ocupacao;Status\n")
        f.write(f"{agora.strftime('%d/%m/%Y')};{agora.strftime('%H:%M:%S')};{tipo};{ocupacao};{status}\n")
    cv2.imwrite(f"capturas/{tipo}_{agora.strftime('%Y%m%d_%H%M%S')}.jpg", frame)

# --- LOOP PRINCIPAL DO SISTEMA ---
def sistema_principal():
    global dados_compartilhados
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    rastreador = Rastreador()

    ret, frame_teste = cap.read()
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    linha_y = altura // 2
    
    entradas, saidas = 0, 0
    limite_maximo = 5
    ids_contados = set()

    # Inicia o servidor Web em uma thread separada
    threading.Thread(target=rodar_servidor, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_limpo = frame.copy()

        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        deteccoes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 3500: deteccoes.append(cv2.boundingRect(cnt))

        objetos = rastreador.atualizar(deteccoes)
        for x, y, w, h, id in objetos:
            cy = y + h // 2
            if id not in ids_contados:
                if cy > (linha_y + 45): # ENTRADA
                    entradas += 1
                    ids_contados.add(id)
                    lotado = (entradas - saidas) >= limite_maximo
                    salvar_dados("ENTRADA", entradas - saidas, frame_limpo, lotado)
                    disparar_alarme("perigo" if lotado else "evento")
                elif cy < (linha_y - 45): # SAÍDA
                    if (entradas - saidas) > 0:
                        saidas += 1
                        ids_contados.add(id)
                        salvar_dados("SAIDA", entradas - saidas, frame_limpo, False)
                        disparar_alarme("evento")
                    else: ids_contados.add(id)

        # ATUALIZA AS VARIÁVEIS GLOBAIS PARA O SITE
        ocupacao_viva = max(0, entradas - saidas)
        dados_compartilhados.update({
            "ocupacao": ocupacao_viva,
            "entradas": entradas,
            "saidas": saidas,
            "status": "ALERTA_LOTADO" if ocupacao_viva >= limite_maximo else "NORMAL"
        })

        # Desenho da interface OpenCV (local)
        cor = (0, 0, 255) if ocupacao_viva >= limite_maximo else (0, 255, 0)
        cv2.line(frame, (0, linha_y), (largura, linha_y), (255, 255, 255), 2)
        cv2.putText(frame, f"WEB Dashboard: ON | Ocupacao: {ocupacao_viva}", (20, 45), 1, 1.2, cor, 2)
        
        cv2.imshow('Vision IA - Passo 10 (Web Server)', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sistema_principal()