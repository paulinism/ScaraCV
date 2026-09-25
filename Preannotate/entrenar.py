"""
entrenar.py
===========
Entrena el modelo YOLO con el dataset generado.

Uso:
    python entrenar.py
"""

from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # se descarga solo la primera vez (~6MB)

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,          # reduce a 4 si tu GPU tiene poca memoria
    device="cpu",         # 0 = GPU   |   "cpu" si no tienes GPU
    name="figuras",
    project="runs/detect",
    patience=10,      # early stopping: para si no mejora en 10 épocas
    pretrained=True,
    verbose=True,
)

print("\n✅ Entrenamiento terminado")
print("   Modelo guardado en: runs/detect/figuras/weights/best.pt")
