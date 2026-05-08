import ultralytics
from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
model = YOLO("yolo11l.pt")

results = model.train(
        data='config.yaml',  # path of the configuration file
        epochs=6000,
        workers=0,
        batch=8 ,            # size of the batch
        imgsz=640,           # each image has 640x640 pixels
        device=0,            # GPU
        pretrained = False,  # Start the model from zero
    )

print("Training completed.")