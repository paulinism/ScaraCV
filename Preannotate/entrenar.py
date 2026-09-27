"""
entrenar.py
===========
Trains the YOLO model with the generated dataset.

"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # downloads automatically the first time (~6MB)

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,          # reduce to 4 if your GPU has low memory