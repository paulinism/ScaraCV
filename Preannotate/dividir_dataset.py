"""
dividir_dataset.py
==================
Divide las imágenes y labels en train/val (80/20)
y genera el data.yaml listo para entrenar con YOLO.

Uso:
    python dividir_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

random.seed(42)

# ── Carpetas de entrada (generadas por auto_annotate.py) ──
IMAGES_DIR = Path("auto_images")
LABELS_DIR = Path("auto_labels")

# ── Carpeta de salida ──
OUTPUT_DIR = Path("dataset")

# ── Proporción train/val ──
TRAIN_RATIO = 0.8

# ── Clases (mismo orden que en auto_annotate.py) ──
CLASSES = [
    "red_square",    "red_triangle",    "red_pentagon",    "red_cross",
    "yellow_square", "yellow_triangle", "yellow_pentagon", "yellow_cross",
    "blue_square",   "blue_triangle",   "blue_pentagon",   "blue_cross",
    "green_square",  "green_triangle",  "green_pentagon",  "green_cross",
]

# ══════════════════════════════════════════════════════════

def main():
    # Crear estructura de carpetas
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Listar imágenes que tienen su .txt correspondiente
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        f for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
        and (LABELS_DIR / f"{f.stem}.txt").exists()
    ]

    if len(images) == 0:
        print("⚠️  No se encontraron imágenes con labels.")
        print(f"   Verifica que existan carpetas '{IMAGES_DIR}' y '{LABELS_DIR}'")
        return

    random.shuffle(images)
    split_idx   = int(len(images) * TRAIN_RATIO)
    train_imgs  = images[:split_idx]
    val_imgs    = images[split_idx:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        for img_path in imgs:
            # Copiar imagen
            shutil.copy(img_path,
                        OUTPUT_DIR / "images" / split / img_path.name)
            # Copiar label
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            shutil.copy(label_path,
                        OUTPUT_DIR / "labels" / split / label_path.name)

    # Generar data.yaml
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names:\n")
        for name in CLASSES:
            f.write(f"  - {name}\n")

    print(f"✅ Dataset creado en '{OUTPUT_DIR}/'")
    print(f"   Train : {len(train_imgs)} imágenes")
    print(f"   Val   : {len(val_imgs)} imágenes")
    print(f"   data.yaml generado")


if __name__ == "__main__":
    main()
