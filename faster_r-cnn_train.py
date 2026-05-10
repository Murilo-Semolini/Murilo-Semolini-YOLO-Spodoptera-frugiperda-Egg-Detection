import os
import cv2
import json
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from tqdm import tqdm


# =========================
# CONFIGURAÇÕES
# =========================

TRAIN_IMAGES = "dataset/train/images"
TRAIN_JSON   = "dataset/train/annotations.json"

VAL_IMAGES   = "dataset/val/images"
VAL_JSON     = "dataset/val/annotations.json"

NUM_CLASSES  = 4       # fundo (0) + suas classes
BATCH_SIZE   = 8
NUM_EPOCHS   = 6000
LEARNING_RATE = 0.0005
NUM_WORKERS  = 0      # ajuste conforme CPUs disponíveis

DEVICE    = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SAVE_PATH = "faster_rcnn_model.pth"


# =========================
# DATASET COCO
# =========================

class CocoDataset(Dataset):

    def __init__(self, images_dir, annotation_file):
        self.images_dir = images_dir

        with open(annotation_file) as f:
            self.coco = json.load(f)

        self.image_data   = self.coco["images"]
        self.annotations  = self.coco["annotations"]

        # image_id -> lista de anotações
        self.annotations_map = {}
        for ann in self.annotations:
            iid = ann["image_id"]
            self.annotations_map.setdefault(iid, []).append(ann)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_info = self.image_data[idx]
        image_id   = image_info["id"]

        image_path = os.path.join(self.images_dir, image_info["file_name"])

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.annotations_map.get(image_id, [])

        boxes  = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            # ignora caixas degeneradas
            if w <= 0 or h <= 0:
                continue

            xmin, ymin = x, y
            xmax, ymax = x + w, y + h

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann["category_id"])

        # FIX: tensor vazio deve ter shape (0, 4), não (0,)
        if len(boxes) == 0:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_tensor  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes_tensor,
            "labels":   labels_tensor,
            "image_id": torch.tensor([image_id]),   # FIX: exigido pela torchvision
        }

        image = F.to_tensor(image)

        return image, target


# =========================
# COLLATE FUNCTION
# =========================

def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# DATASETS E DATALOADERS
# =========================

train_dataset = CocoDataset(TRAIN_IMAGES, TRAIN_JSON)
val_dataset   = CocoDataset(VAL_IMAGES,   VAL_JSON)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# =========================
# MODELO
# =========================

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )
)

model.to(DEVICE)


# =========================
# OTIMIZADOR E SCHEDULER
# =========================

# FIX: SGD com momentum é o padrão para detecção de objetos
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005,
)

# FIX: scheduler reduz LR nas épocas 10 e 16
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[10, 16],
    gamma=0.1,
)


# =========================
# VALIDAÇÃO COM mAP
# =========================

def evaluate(model, data_loader, device):
    """
    Roda o modelo em modo de inferência e calcula mAP usando torchmetrics.
    Retorna um dicionário com as métricas (map, map_50, map_75, etc.).
    """
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validando"):
            images = [img.to(device) for img in images]

            outputs = model(images)

            # torchmetrics espera listas de dicts com tensors na CPU
            preds = [
                {
                    "boxes":  out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                }
                for out in outputs
            ]

            gts = [
                {
                    "boxes":  t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
                for t in targets
            ]

            metric.update(preds, gts)

    return metric.compute()


# =========================
# LOOP DE TREINAMENTO
# =========================

best_map = 0.0

for epoch in range(NUM_EPOCHS):

    # ---------- TREINO ----------
    model.train()

    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [treino]")

    for images, targets in progress_bar:

        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value  = losses.item()
        epoch_loss += loss_value

        progress_bar.set_postfix(loss=f"{loss_value:.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} | Loss médio: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()}")

    lr_scheduler.step()   # FIX: atualiza o LR após cada época

    # ---------- VALIDAÇÃO ----------
    metrics = evaluate(model, val_loader, DEVICE)

    map_val    = metrics["map"].item()
    map_50_val = metrics["map_50"].item()
    map_75_val = metrics["map_75"].item()

    print(
        f"Validação | mAP: {map_val:.4f} | "
        f"mAP@50: {map_50_val:.4f} | mAP@75: {map_75_val:.4f}"
    )

    # salva o melhor modelo
    if map_val > best_map:
        best_map = map_val
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  → Melhor modelo salvo (mAP={best_map:.4f})")

    print("-" * 60)


print(f"\nTreinamento concluído. Melhor mAP: {best_map:.4f}")
print(f"Modelo salvo em: {SAVE_PATH}")