from ultralytics import YOLO
model = YOLO("C:/Users/IFSP SRT/PycharmProjects/muriloIC/runs/detect/train3/weights/best.pt")

results = model("C:/Users/IFSP SRT/PycharmProjects/muriloIC/Imagens_Dificil/Postura 1 (20230509).jpg", conf=0.6)
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

print(f"Total de ovos na imagem é aproximadamente: {int(total)}")