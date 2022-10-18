import face_recognition as fr # importa a biblioteca face_recognition e permite que ela seja acessada por meio do objeto 'fr'

# Função para Reconhecer Face
def reconhece_face(url_foto):
    foto = fr.load_image_file(url_foto) # Carrega a imagem do rosto 
    rostos = fr.face_encodings(foto) # Processo necessario para comparação e verificação de existencia de rosto
    if(len(rostos) > 0): # Loop que verifica cada item dentro do vetor
        return True, rostos #  Se houver algum rosto retorna TRUE e os rostos encontrados
    
    return False, [] # Se não houver algum rosto retorna FALSE

# Função para retornar rostos cadastrados
def get_rostos():
    rostos_conhecidos = [] # instância uma lista para posteriormente adicionar rostos_conhecidos
    nomes_dos_rostos = [] # Instância uma lista para posteriormente adicionar nomes

    medina = reconhece_face("./img/medina.jpg") # Armazena na variavel 'medina' o retorno da função 'reconhece_face' que esta acima
    if(medina[0]): # IF para verificar se o retorno da função 'reconhece_face' caso seja TRUE entrara na condição
        rostos_conhecidos.append(medina[1][0]) # Adiciona rosto retornado da função 'reconhece_face' na lista de rostos conhecidos
        nomes_dos_rostos.append("Medina Lindo") # Adiciona um Nome na lista 'nomes_dos_rostos'

    # Processo se repete para cadastro de outro rosto
    william = reconhece_face("./img/william.jpg") # Armazena na variavel 'william' o retorno da função 'reconhece_face' que esta acima
    if(william[0]): # IF para verificar se o retorno da função 'reconhece_face' caso seja TRUE entrara na condição
        rostos_conhecidos.append(william[1][0]) # Adiciona rosto retornado da função 'reconhece_face' na lista de rostos conhecidos
        nomes_dos_rostos.append("William") # Adiciona um Nome na lista 'nomes_dos_rostos'

    # Processo se repete para cadastro de outro rosto
    fabiano = reconhece_face("./img/fabiano.jpg") # Armazena na variavel 'william' o retorno da função 'reconhece_face' que esta acima
    if(fabiano[0]): # IF para verificar se o retorno da função 'reconhece_face' caso seja TRUE entrara na condição
        rostos_conhecidos.append(fabiano[1][0]) # Adiciona rosto retornado da função 'reconhece_face' na lista de rostos conhecidos
        nomes_dos_rostos.append("fabiano") # Adiciona um Nome na lista 'nomes_dos_rostos'
    
    return rostos_conhecidos, nomes_dos_rostos # Retorna as duas listas instânciadas no inicio da função
    