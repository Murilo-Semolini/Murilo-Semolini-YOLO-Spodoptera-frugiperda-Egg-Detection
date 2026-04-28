import ultralytics
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
model = YOLO("yolo11l.pt")

results = model.train(
        data='config.yaml',  # Caminho do arquivo
        epochs=6000,
        workers=0,
        batch=8 ,            # Tinha tentando com 64, mas acabou a RAM da GPU
        imgsz=640,           # 640px
        device=0,             # GPU
        pretrained = False,  # Começar um modelo do zero
    )

print("Treinamento concluído.")