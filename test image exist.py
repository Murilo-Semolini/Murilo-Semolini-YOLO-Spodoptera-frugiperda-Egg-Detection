import os
IMAGE_PATH = "C:\\Users\\semol\\Documents\\faculdade 2025\\Iniciação científica\\redes e códigos\\dataset coco com split de testes\\train\\images\\Postura-3-20230516-_jpg.rf.36cf4d9b20a1b44de30ae9292041f9bb.jpg"
print("Arquivo existe?", os.path.exists(IMAGE_PATH))
print("Caminho absoluto:", os.path.abspath(IMAGE_PATH))