import numpy as np # importa a biblioteca 'numpy' e permite que ela seja acessada por meio do objeto 'fr'
import face_recognition as fr # importa a biblioteca face_recognition e permite que ela seja acessada por meio do objeto 'fr'
import cv2 # importa a biblioteca cv2
from engine import get_rostos # importa as funções criadas no arquivo "engine.py"

rostos_conhecidos, nomes_dos_rostos = get_rostos() # Recebe como retorno da função "get_rostos" os rostos e nomes cadastrados

video_capture = cv2.VideoCapture(0) # A variavel "video_capture" recebe qual é a camera principal do computador para realizar a analise do rosto posteriormente
while True: # Loop infinito

    # ret é uma variável booleana que retorna true se o quadro estiver disponível.
    # frame é um vetor de matriz de imagem capturado com base nos quadros por segundo padrão definidos explicitamente ou implicitamente
    ret, frame = video_capture.read() # As variaveis recebem o retorno da função "video_capture.read()" que é responsavel pela captura de video

    rgb_frame = frame[:, :, ::-1] # Linha responsavel por dar cor ao frame que ficara no rosto do usuario

    localizacao_dos_rostos = fr.face_locations(rgb_frame) # A variavel "localizacao_dos_rostos" recebe as coordenadas das faces 
    rosto_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos) # Recebe as faces contidas na camera

    for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos): # Inicio do loop
        resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido) # Armazena na variavel "resultados" um valor booleano resultante da comparação dos rostos cadastrados com os contidos na camera
        print(resultados) # Mostra no terminal a comparação - afim de debugar

        face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido) # O método "face_distance" retorna o quão semelhantes são as faces. e retorna um ndarray numpy com a distância para cada face
        
        melhor_id = np.argmin(face_distances) # A variavel "melhor_id" recebe o menor valor inteiro dentro do array "face_distances"
        if resultados[melhor_id]: # IF entrara na condição se as codificações das faces contidas na linha 21 se corresponderem
            nome = nomes_dos_rostos[melhor_id] # variavel nome recebe um nome contido na lista "nomes_dos_rostos" com base no valor da variavel "melhor_id"
        else: # Entrara no ELSE caso as codificações das faces contidas na linha 21 não se correspondam indicando um rosto não cadastrado
            nome = "Desconhecido" # variavel "nome" recebe uma string "Desconhecido"
        
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1) # Estiliza o retangulo colocado ao redor do rosto

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED) # Estiliza a parte de baixo do retangulo onde se encontram os nomes
        font = cv2.FONT_HERSHEY_SIMPLEX # Define a fonte a ser utilizada

        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 0.50, (255, 255, 255), 1) # Coloca o texto no retangulo que se encontra em baixo do rosto

        cv2.imshow('Webcam_facerecognition', frame) # Linha responsavel por abrir a janela com a camera

    if cv2.waitKey(1) & 0xFF == ord('q'): # Coloca como tecla de escape o "q" sendo responsavel por parar o loop
        break # para o loop

video_capture.release() # Encerra o uso da camera em outros aplicativos para evitar conflitos
cv2.destroyAllWindows() # Fecha a janela da camera ao imterromper o loop

# https://face-recognition.readthedocs.io/en/latest/face_recognition.html?highlight=face_distance#face_recognition.api.face_distance
# https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html
# https://numpy.org/doc/stable/reference/generated/numpy.argmin.html