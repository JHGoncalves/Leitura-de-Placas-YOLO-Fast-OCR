Sistema de Reconhecimento de Placas de Veículos em Python
1. Sobre o Projeto
Este projeto é uma aplicação de visão computacional que realiza a detecção e o reconhecimento de placas de veículos a partir de arquivos de vídeo. A pipeline consiste em:

Detecção: Utiliza um modelo YOLOv8 para localizar as coordenadas das placas em cada quadro do vídeo.

Pré-processamento: Aplica uma transformação de perspectiva na placa detectada para normalizar sua orientação e melhorar a acurácia do OCR.

Reconhecimento (OCR): Extrai os caracteres alfanuméricos da placa processada usando a biblioteca Fast-Plate-OCR.

Validação: Compara a placa lida com um valor de referência e gera um vídeo de saída com as detecções e o status da validação (Autorizado/Não Autorizado) desenhados em tempo real.

2. Tecnologias Principais
Python 3.9+

YOLOv8 (Ultralytics)

Fast-Plate-OCR

OpenCV

NumPy

3. Configuração do Ambiente
Este guia assume que você já clonou o repositório e está na pasta raiz do projeto.

a. Crie e Ative um Ambiente Virtual

É fortemente recomendado executar o projeto em um ambiente virtual para gerenciar as dependências de forma isolada.

# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente
# No Linux/macOS:
source .venv/bin/activate
# No Windows:
# .venv\Scripts\activate

b. Instale as Dependências

Todas as bibliotecas necessárias estão listadas no arquivo requirements.txt. Para instalá-las, execute:

pip install -r requirements.txt

4. Como Utilizar
A execução do projeto é controlada por meio de variáveis de configuração no script principal.

a. Posicione seu Vídeo de Entrada

Coloque o arquivo de vídeo que deseja processar dentro do diretório /videos.

b. Configure o Script

Abra o arquivo app/reconhecimento.py e ajuste as seguintes variáveis no topo do script:

CAMINHO_VIDEO_TESTE: O caminho relativo para o seu vídeo de entrada.
CAMINHO_VIDEO_TESTE = "videos/seu-video.mp4"

CAMINho_VIDEO_SAIDA: O caminho relativo onde o vídeo de saída processado será salvo.
CAMINHO_VIDEO_SAIDA = "videos/resultado-seu-video.mp4"

PLACA_CADASTRADA: A string da placa a ser utilizada como referência para a validação.
PLACA_CADASTRADA = "ABC1D23"

c. Execute a Aplicação

Após salvar suas configurações no script, execute a aplicação a partir do diretório raiz do projeto:

python app/reconhecimento.py