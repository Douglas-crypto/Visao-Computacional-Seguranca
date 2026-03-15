# 📑 Planejamento & Arquitetura: Vision IA 2026

## 1. O Problema
**O que quero resolver?**
Monitorar e contar objetos/pessoas em tempo real usando Visão Computacional, resolvendo a necessidade de controle de lotação em espaços físicos de forma automatizada.

**Por que?**
Estudar a integração de modelos de IA e processamento de imagem com persistência de dados e automação no Linux, criando uma solução que seja leve, escalável e de fácil consulta via Web.

## 2. Requisitos Iniciais (Hardware & Software)
* **SO:** Linux (Ubuntu/Debian com suporte a X11/Wayland).
* **Hardware:** Webcam integrada ou USB.
* **Linguagem:** Python 3.12+
* **Bibliotecas Base:** * `OpenCV`: Processamento de imagem.
    * `Flask`: Interface Web/Dashboard.
    * `Threading`: Processamento paralelo.

## 3. Arquitetura do Sistema (Final)
Para garantir que o processamento da câmera não sofra atrasos por conta da interface web, implementamos uma arquitetura **Multithreading**:

1.  **Thread de Visão:** Captura -> Processamento (MOG2) -> Lógica de Contagem -> Logs.
2.  **Thread de Servidor:** Flask servindo o Dashboard em `0.0.0.0:5000`.
3.  **Thread de Monitoramento:** Verifica se a porta 5000 está ativa e dispara o navegador automaticamente.

## 4. Fluxo de Dados Realizado
1. **Captura:** Leitura de frames via OpenCV.
2. **Tratamento:** Aplicação de máscaras de fundo e filtros de contorno.
3. **Rastreamento:** Identificação de IDs únicos para evitar contagens duplicadas.
4. **Persistência:** Registro de eventos em CSV e salvamento de fotos em caso de entrada/saída.
5. **Dashboard:** Atualização em tempo real das variáveis globais consumidas pelo Flask.