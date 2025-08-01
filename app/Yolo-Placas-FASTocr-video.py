import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#### Funções para imagens

def preparar_imagem_para_ocr(imagem_original, box_yolo_numpy):
    """Aplica correção de perspectiva para melhorar a leitura do OCR."""
    try:
        x1, y1, x2, y2 = box_yolo_numpy[:4].astype(int)
        src_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype="float32")
        largura_saida = 224
        altura_saida = 70
        dst_pts = np.array([[0, 0], [largura_saida - 1, 0],
                            [largura_saida - 1, altura_saida - 1], [0, altura_saida - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        placa_alinhada = cv2.warpPerspective(imagem_original, matrix, (largura_saida, altura_saida))
        return placa_alinhada
    except Exception as e:
        print(f"Erro no pré-processamento da placa: {e}")
        return None

def limpar_texto_placa(texto):
    """Remove caracteres especiais e formata para maiúsculo."""
    return re.sub(r'[^A-Z0-9]', '', texto).upper()



#### Funções para vídeo

def processar_video(caminho_video, placa_cadastrada, caminho_saida):
    """Função principal que processa um vídeo e verifica as placas em cada quadro."""
    try:
        # Abre o vídeo de entrada
        cap = cv2.VideoCapture(caminho_video)
        if not cap.isOpened():
            print(f"❌ ERRO: Não foi possível abrir o vídeo em '{caminho_video}'")
            return

        # Pega as propriedades do vídeo para criar o arquivo de saída
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define o codec e cria o objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para .mp4
        out = cv2.VideoWriter(caminho_saida, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Fim do vídeo

            # 1. Detecção com YOLO
            resultados_yolo = model_yolo(frame)[0]

            # Itera sobre cada placa detectada no quadro
            for box in resultados_yolo.boxes:
                # Pega as coordenadas da caixa delimitadora
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 2. Prepara a imagem da placa para o OCR
                placa_alinhada = preparar_imagem_para_ocr(frame, box.xyxy[0].cpu().numpy())
                
                if placa_alinhada is not None:
                    # 3. Reconhecimento com Fast-Plate-OCR
                    resultado_ocr = model_ocr.run(placa_alinhada)
                    texto_bruto_ocr = resultado_ocr[0] if resultado_ocr else ""
                    placa_processada = limpar_texto_placa(texto_bruto_ocr)

                    # 4. Verificação e desenho no quadro
                    autorizado = (placa_processada == placa_cadastrada)
                    cor_texto = (0, 255, 0) if autorizado else (0, 0, 255) # Verde se autorizado, Vermelho se não
                    
                    # Desenha o retângulo ao redor da placa
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor_texto, 3)
                    # Escreve o texto da placa acima do retângulo
                    cv2.putText(frame, placa_processada, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor_texto, 2)
            
            # Escreve o quadro processado no arquivo de saída
            out.write(frame)


        # Libera os recursos
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Vídeo processado com sucesso e salvo em: '{caminho_saida}'")

    except Exception as e:
        print(f"❌ ERRO INESPERADO DURANTE O PROCESSAMENTO DO VÍDEO: {e}")


# --- CONFIGURAÇÕES PARA VÍDEO ---
CAMINHO_VIDEO_TESTE = "../videos/video-carro.mp4" # Coloque o nome do seu arquivo de vídeo aqui
CAMINHO_VIDEO_SAIDA = "../videos_saida/resultado-video.mp4" # Diretório do video resultado
PLACA_CADASTRADA = "XXXXXXX" # Placa a ser validada

# 4. Bloco de execução principal
if __name__ == "__main__":
    print("Carregando modelos...")
    # Lógica para carregar os modelos
    model_yolo = YOLO("license-plate-finetune-v1x.pt")
    model_ocr = LicensePlateRecognizer("cct-xs-v1-global-model")
    print("Modelos carregados.")

    # Chama a função principal que faz o trabalho
    processar_video(CAMINHO_VIDEO_TESTE, PLACA_CADASTRADA, CAMINHO_VIDEO_SAIDA)