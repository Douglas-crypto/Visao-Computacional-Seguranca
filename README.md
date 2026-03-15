# Vision IA - Sistema de Monitoramento e Contagem

Sistema de Visão Computacional de alta performance desenvolvido para monitoramento de fluxo de pessoas/objetos em tempo real, integrando processamento de imagem e interface web.

## Funcionalidades Principais

* **Contagem Bidirecional:** Identifica entradas e saídas através de uma linha virtual inteligente.
* **Dashboard Web Real-time:** Interface moderna via Flask para acompanhamento remoto.
* **Arquitetura Multithreading:** Garante que a captura da câmera e o servidor web rodem de forma independente e sem atrasos.
* **Alertas Inteligentes:** Notificações por voz (Linux `spd-say`) e alarmes sonoros configuráveis.
* **Persistência de Dados:** Geração automática de logs em CSV e capturas de tela dos eventos detectados.

## Stack Tecnológica
* **Linguagem:** Python 3.12
* **Visão Computacional:** OpenCV (MOG2 Background Subtraction + Centroid Tracking)
* **Web Framework:** Flask
* **Distribuição:** PyInstaller (Executável Standalone)

## Como Usar

### 1. Executável (Recomendado)
Para rodar sem precisar instalar Python, acesse a aba **Releases** deste repositório, baixe o binário e execute:
```bash
chmod +x VisionIA
./VisionIA
