"""
auto_annotate.py
================
Pre-anota automáticamente todas las imágenes de un ZIP usando segmentación HSV.
Genera archivos .txt en formato YOLO para cada imagen.

Uso:
    python auto_annotate.py --zip imagenes.zip
    python auto_annotate.py --folder ./mis_fotos

Resultado:
    carpeta  auto_labels/  con un .txt por imagen
    carpeta  auto_preview/ con imágenes anotadas para verificar visualmente
"""

import cv2
import numpy as np
import zipfile
import argparse
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CLASES — deben coincidir EXACTAMENTE con data.yaml después
# ═══════════════════════════════════════════════════════════════════════════════
# Índice : nombre
CLASSES = {
    0:  "red_square",
    1:  "red_triangle",
    2:  "red_pentagon",
    3:  "red_cross",
    4:  "yellow_square",
    5:  "yellow_triangle",
    6:  "yellow_pentagon",
    7:  "yellow_cross",
    8:  "blue_square",
    9:  "blue_triangle",
    10: "blue_pentagon",
    11: "blue_cross",
    12: "green_square",
    13: "green_triangle",
    14: "green_pentagon",
    15: "green_cross",
}

# Nombre → índice (inverso)
CLASS_INDEX = {v: k for k, v in CLASSES.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# RANGOS HSV POR COLOR
# Si los colores no se detectan bien, ajusta los valores aquí.
# Puedes probar con la imagen de preview generada.
# ═══════════════════════════════════════════════════════════════════════════════
COLOR_RANGES = {
    "red": [
        (np.array([0,   130,  60]),  np.array([10,  255, 255])),
        (np.array([168, 130,  60]),  np.array([180, 255, 255])),
    ],
    "yellow": [
        (np.array([18,  100,  80]),  np.array([38,  255, 255])),
    ],
    "green": [
        (np.array([38,  80,   40]),  np.array([88,  255, 255])),
    ],
    "blue": [
        (np.array([95,  80,   40]),  np.array([135, 255, 255])),
    ],
}

# Área mínima en píxeles — filtra ruido pequeño
MIN_AREA = 2000
# Área máxima — filtra el robot y objetos grandes
MAX_AREA = 90_000

# ═══════════════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE FORMA POR GEOMETRÍA
# ═══════════════════════════════════════════════════════════════════════════════

def classify_shape(contour):
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
    n      = len(approx)

    if n == 3:
        return "triangle"
    if n == 4:
        return "square"
    if n == 5:
        return "pentagon"

    # Cruz: área del contorno mucho menor que su convex hull
    hull_area    = cv2.contourArea(cv2.convexHull(contour))
    contour_area = cv2.contourArea(contour)
    if hull_area > 0 and (contour_area / hull_area) < 0.72:
        return "cross"
    if n >= 8:
        return "cross"

    return None   # desconocido → ignorar


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESAR UNA IMAGEN → lista de anotaciones YOLO
# ═══════════════════════════════════════════════════════════════════════════════

def annotate_image(image):
    """
    Devuelve lista de (class_id, cx_norm, cy_norm, w_norm, h_norm)
    Todo normalizado entre 0 y 1 como requiere YOLO.
    """
    h_img, w_img = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    annotations = []
    seen_centers = []

    for color_name, ranges in COLOR_RANGES.items():
        # Construir máscara del color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # Limpiar ruido morfológico
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_AREA < area < MAX_AREA):
                continue

            shape_name = classify_shape(cnt)
            if shape_name is None:
                continue

            class_name = f"{color_name}_{shape_name}"
            class_id   = CLASS_INDEX.get(class_name)
            if class_id is None:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Centro normalizado
            cx = (x + w / 2.0) / w_img
            cy = (y + h / 2.0) / h_img
            wn = w / w_img
            hn = h / h_img

            # Evitar duplicados (mismo centro ± 40px)
            dup = any(abs((x + w/2) - sx) < 40 and abs((y + h/2) - sy) < 40
                      for sx, sy in seen_centers)
            if dup:
                continue

            seen_centers.append((x + w/2, y + h/2))
            annotations.append((class_id, cx, cy, wn, hn))

    return annotations


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGEN DE PREVIEW — para verificar visualmente que las anotaciones están bien
# ═══════════════════════════════════════════════════════════════════════════════

COLOR_BGR = {
    "red":    (0,   0,   220),
    "yellow": (0,   210, 210),
    "green":  (0,   200, 0),
    "blue":   (220, 80,  0),
}

def draw_preview(image, annotations):
    preview = image.copy()
    h_img, w_img = image.shape[:2]

    for (class_id, cx, cy, wn, hn) in annotations:
        class_name = CLASSES[class_id]
        color_part = class_name.split("_")[0]
        bgr        = COLOR_BGR.get(color_part, (200, 200, 200))

        # Desnormalizar
        x1 = int((cx - wn / 2) * w_img)
        y1 = int((cy - hn / 2) * h_img)
        x2 = int((cx + wn / 2) * w_img)
        y2 = int((cy + hn / 2) * h_img)

        cv2.rectangle(preview, (x1, y1), (x2, y2), bgr, 3)
        cv2.putText(preview, class_name, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    return preview


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDAR .txt EN FORMATO YOLO
# ═══════════════════════════════════════════════════════════════════════════════

def save_yolo_txt(annotations, txt_path):
    with open(txt_path, "w") as f:
        for (class_id, cx, cy, wn, hn) in annotations:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESAR ZIP
# ═══════════════════════════════════════════════════════════════════════════════

def process_zip(zip_path):
    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    labels_dir  = Path("auto_labels")
    preview_dir = Path("auto_preview")
    images_dir  = Path("auto_images")

    labels_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    total = 0
    total_boxes = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = [f for f in zf.namelist()
                 if Path(f).suffix.lower() in IMAGE_EXTS
                 and not f.startswith("__MACOSX")]

        print(f"📦 {len(files)} imágenes encontradas en el ZIP\n")

        for fname in files:
            with zf.open(fname) as f:
                buf   = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

            if image is None:
                continue

            stem        = Path(fname).stem
            annotations = annotate_image(image)

            # Guardar imagen original
            cv2.imwrite(str(images_dir / Path(fname).name), image)

            # Guardar .txt YOLO
            save_yolo_txt(annotations, labels_dir / f"{stem}.txt")

            # Guardar preview
            preview = draw_preview(image, annotations)
            cv2.imwrite(str(preview_dir / f"{stem}_preview.jpg"), preview)

            print(f"  ✓ {Path(fname).name:40s}  {len(annotations)} figuras")
            total += 1
            total_boxes += len(annotations)

    print(f"\n{'═'*55}")
    print(f"  Imágenes procesadas : {total}")
    print(f"  Bounding boxes      : {total_boxes}")
    print(f"  Labels guardados en : auto_labels/")
    print(f"  Previews en         : auto_preview/")
    print(f"  Imágenes en         : auto_images/")
    print(f"{'═'*55}")
    print(f"\n✅ Siguiente paso: revisa las imágenes en auto_preview/")
    print(f"   y corrige errores en LabelImg apuntando a auto_images/ + auto_labels/")


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESAR CARPETA
# ═══════════════════════════════════════════════════════════════════════════════

def process_folder(folder_path):
    IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder      = Path(folder_path)
    labels_dir  = Path("auto_labels")
    preview_dir = Path("auto_preview")

    labels_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    files = [f for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    print(f"📁 {len(files)} imágenes encontradas\n")

    total_boxes = 0
    for img_path in sorted(files):
        image       = cv2.imread(str(img_path))
        annotations = annotate_image(image)

        save_yolo_txt(annotations, labels_dir / f"{img_path.stem}.txt")
        preview = draw_preview(image, annotations)
        cv2.imwrite(str(preview_dir / f"{img_path.stem}_preview.jpg"), preview)

        print(f"  ✓ {img_path.name:40s}  {len(annotations)} figuras")
        total_boxes += len(annotations)

    print(f"\n✅ {total_boxes} bounding boxes generados")
    print(f"   Labels  → auto_labels/")
    print(f"   Preview → auto_preview/")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-anotación automática de figuras geométricas para YOLO"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip",    help="Ruta al archivo .zip con imágenes")
    group.add_argument("--folder", help="Ruta a carpeta con imágenes")
    args = parser.parse_args()

    if args.zip:
        process_zip(args.zip)
    else:
        process_folder(args.folder)
