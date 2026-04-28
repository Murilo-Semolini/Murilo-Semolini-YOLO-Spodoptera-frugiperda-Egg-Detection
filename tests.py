from ultralytics import YOLO
model = YOLO("C:/Users/IFSP SRT/PycharmProjects/muriloIC/runs/detect/train9/weights/best.pt")

imagens = ["C:/Users/IFSP SRT/PycharmProjects/muriloIC/datasetV11/valid/images/Postura-16-20230520-_jpg.rf.2f9b088280be0e183989491b134e01fd.jpg",
           "C:/Users/IFSP SRT/PycharmProjects/muriloIC/datasetV11/valid/images/Postura-3-20230515-_jpg.rf.67e1ceaf41ca54605784857b7558f6df.jpg",
           "C:/Users/IFSP SRT/PycharmProjects/muriloIC/datasetV11/valid/images/3_camadas_0_jpg.rf.a44ff0ade7aa71c319168ff02db58748.jpg"]

imagem_atual = 1

for img in imagens:

    results = model(img, conf=0.5)
    results[0].show()

    total = 0

    for caixa in results[0].boxes:
        x1, y1, x2, y2 = caixa.xyxy[0]
        conf = caixa.conf[0]
        classe = caixa.cls[0]
        if(classe == 0):
            total = total + 2
        elif(classe == 1):
            total = total + 3
        else:
            total = total + 1


    print(f"Total de ovos na imagem {imagem_atual} é aproximadamente: {int(total)}")
    imagem_atual = imagem_atual + 1