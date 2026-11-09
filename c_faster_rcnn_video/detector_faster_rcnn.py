import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)


def main():
    parser = argparse.ArgumentParser(description='Actividad 06 C - Faster R-CNN en video')
    parser.add_argument('--video', default='0', help='Indice de webcam (0) o ruta a video')
    parser.add_argument('--conf', type=float, default=0.60, help='Umbral de confianza (0-1)')
    parser.add_argument('--width', type=int, default=640, help='Ancho de procesamiento')
    parser.add_argument('--guardar', action='store_true', help='Guardar video con detecciones')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf debe estar entre 0 y 1')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Dispositivo: {device}')
    print('Cargando Faster R-CNN con pesos COCO (requiere internet la primera vez)...')
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device).eval()
    preprocess = weights.transforms()
    classes = weights.meta['categories']

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir la fuente de video: {args.video}')

    Path('resultados').mkdir(exist_ok=True)
    writer = None
    csv_file = open('resultados/metricas.csv', 'w', newline='', encoding='utf-8')
    log = csv.writer(csv_file)
    log.writerow(['fotograma', 'latencia_ms', 'fps_procesamiento', 'num_detecciones'])
    frame_id, last_show = 0, time.perf_counter()
    print('Controles: Q = salir, S = guardar captura')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            h, w = frame.shape[:2]
            target_w = min(args.width, w)
            target_h = max(1, round(h * target_w / w))
            frame = cv2.resize(frame, (target_w, target_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess(torch.from_numpy(rgb).permute(2, 0, 1)).to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model([tensor])[0]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latency_ms = elapsed * 1000
            fps_process = 1 / max(elapsed, 1e-9)

            scores = prediction['scores'].cpu().numpy()
            boxes = prediction['boxes'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            kept = 0
            for box, label, score in zip(boxes, labels, scores):
                if score < args.conf:
                    continue
                kept += 1
                x1, y1, x2, y2 = np.asarray(box, dtype=int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 90), 2)
                name = classes[int(label)]
                caption = f'{name}: {score:.0%}'
                cv2.putText(frame, caption, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 90), 2)

            now = time.perf_counter()
            fps_display = 1 / max(now - last_show, 1e-9)
            last_show = now
            cv2.rectangle(frame, (0, 0), (min(410, target_w), 77), (20, 20, 20), -1)
            cv2.putText(frame, f'Faster R-CNN | {device.type.upper()}', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .59, (255, 255, 255), 2)
            cv2.putText(frame, f'Inferencia: {latency_ms:.1f} ms | FPS: {fps_display:.1f}', (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, .51, (255, 255, 255), 1)
            cv2.putText(frame, f'Objetos: {kept} | Conf: {args.conf:.0%}', (8, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, .51, (255, 255, 255), 1)
            log.writerow([frame_id, round(latency_ms, 2), round(fps_display, 2), kept])

            if args.guardar:
                if writer is None:
                    writer = cv2.VideoWriter('resultados/deteccion.mp4',
                                             cv2.VideoWriter_fourcc(*'mp4v'), 15,
                                             (target_w, target_h))
                    if not writer.isOpened():
                        raise RuntimeError('No se pudo crear el video de salida')
                writer.write(frame)
            cv2.imshow('Actividad 06 C - Faster R-CNN', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                file = f'resultados/captura_{frame_id:05d}.jpg'
                cv2.imwrite(file, frame)
                print(f'Captura guardada: {file}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()
    print('Listo. Metricas guardadas en resultados/metricas.csv')


if __name__ == '__main__':
    main()
