import torch
import torchvision
import torchvision.transforms.functional as F
import cv2
import numpy as np

# =========================
# CONFIGURAÇÕES
# =========================
DEVICE     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAVE_PATH  = "C:\\Users\\semol\\Documents\\faculdade 2025\\Iniciação científica\\redes e códigos\\faster_rcnn_model.pth"
IMAGE_PATH = r"C:\Users\semol\Documents\faculdade 2025\Iniciação científica\redes e códigos\dataset coco com split de testes\test\images\Postura-5-20230518-_jpg.rf.b7dbe7d4527f2c462b56ad3887ee7746.jpg"
NUM_CLASSES = 4
SCORE_THRESHOLD = 0.4               # ignora detecções com confiança abaixo disso

CATEGORY_NAMES = {
    1: "2 eggs",
    2: "3 eggs",
    3: "egg",
}

# =========================
# CARREGANDO O MODELO
# =========================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# CARREGANDO E PREPARANDO A IMAGEM
# =========================
with open(IMAGE_PATH, "rb") as f:
    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image_bgr is None:
    raise ValueError("OpenCV não conseguiu decodificar a imagem.")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_tensor = F.to_tensor(image_rgb).to(DEVICE)

# =========================
# INFERÊNCIA
# =========================
with torch.no_grad():
    outputs = model([image_tensor])

output = outputs[0]
boxes  = output["boxes"].cpu().numpy()
labels = output["labels"].cpu().numpy()
scores = output["scores"].cpu().numpy()

# =========================
# DESENHANDO AS CAIXAS
# =========================
result = image_bgr.copy()

for box, label, score in zip(boxes, labels, scores):
    if score < SCORE_THRESHOLD:
        continue                    # ignora detecções fracas

    x1, y1, x2, y2 = map(int, box)
    name = CATEGORY_NAMES.get(label, f"classe {label}")
    text = f"{name}: {score:.2f}"

    # desenha a caixa e o texto
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.putText(result, text, (x1, y1 - 8),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# =========================
# SALVANDO O RESULTADO
# =========================
OUTPUT_PATH = "resultado.jpg"
cv2.imwrite(OUTPUT_PATH, result)
print(f"Imagem salva em: {OUTPUT_PATH}")
print(f"Total de detecções (score ≥ {SCORE_THRESHOLD}): {(scores >= SCORE_THRESHOLD).sum()}")