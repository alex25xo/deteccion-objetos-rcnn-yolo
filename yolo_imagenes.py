"""
Parte b) Detección de objetos con YOLO en imágenes
Autor: Maligno
Uso:
    pip install ultralytics opencv-python pandas
    python yolo_imagenes.py --fuente imagenes/ --modelo yolo11n.pt --map
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detectar(modelo, carpeta, salida, conf, imgsz=640):
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    imagenes = [p for p in sorted(Path(carpeta).iterdir()) if p.suffix.lower() in EXT]
    if not imagenes:
        raise SystemExit(f"No se encontraron imágenes en {carpeta}")

    # Calentamiento (warmup)
    modelo.predict(imagenes[0], imgsz=imgsz, verbose=False)

    for ruta in imagenes:
        t0 = time.perf_counter()
        r = modelo.predict(ruta, conf=conf, imgsz=imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        confs = r.boxes.conf.tolist()
        clases = [modelo.names[int(c)] for c in r.boxes.cls.tolist()]
        
        # Guardar imagen con cajas anotadas
        cv2.imwrite(str(salida / ruta.name), r.plot())

        filas.append({
            "imagen": ruta.name,
            "tiempo_ms": round(ms, 2),
            "n_detecciones": len(confs),
            "conf_media": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "clases": ", ".join(sorted(set(clases))),
        })
        print(f"{ruta.name}: {len(confs)} objetos, {ms:.1f} ms")

    df = pd.DataFrame(filas)
    
    # Guardar en raíz o en carpeta de salida (se guarda en salida por orden)
    csv_path = salida / "resultados_yolo_imagenes.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n=== Resumen ===")
    print(f"Tiempo medio por imagen: {df.tiempo_ms.mean():.2f} ms")
    print(f"FPS equivalente:         {1000 / df.tiempo_ms.mean():.2f}")
    print(f"Detecciones promedio:    {df.n_detecciones.mean():.2f}")
    print(f"Confianza media:         {df.conf_media.mean():.3f}")
    print(f"Resultados guardados en: {csv_path}")


def evaluar_map(modelo, imgsz=640):
    """mAP sobre COCO128 (se descarga solo, ~7 MB)."""
    print("\nEvaluando mAP en COCO128...")
    m = modelo.val(data="coco128.yaml", imgsz=imgsz, verbose=False)
    print("\n=== Métricas de precisión (COCO128) ===")
    print(f"Precision:    {m.box.mp:.3f}")
    print(f"Recall:       {m.box.mr:.3f}")
    print(f"mAP@0.50:     {m.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {m.box.map:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="imagenes", help="Carpeta de imágenes de entrada")
    ap.add_argument("--modelo", default="yolo11n.pt", help="Pesos de YOLO")
    ap.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    ap.add_argument("--imgsz", type=int, default=640, help="Resolución de inferencia")
    ap.add_argument("--salida", default="salida_yolo_imagenes", help="Carpeta de salida")
    ap.add_argument("--map", action="store_true", help="Calcular mAP en COCO128")
    a = ap.parse_args()

    yolo = YOLO(a.modelo)
    detectar(yolo, a.fuente, Path(a.salida), a.conf, a.imgsz)
    if a.map:
        evaluar_map(yolo, a.imgsz)