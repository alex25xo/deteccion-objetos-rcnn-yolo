## b) Detección de objetos con YOLO en imágenes

**Responsable:** Maligno

### Enfoque
YOLO (You Only Look Once) es un detector de **una sola etapa**: una única red convolucional predice cajas y clases en un solo paso hacia adelante, sin requerir una etapa previa de generación de propuestas de región. Se utilizó el modelo liviano `yolo11n` (Ultralytics) preentrenado sobre el conjunto de datos COCO (80 clases comunes).

### Ejecución
```bash
pip install ultralytics opencv-python pandas
python yolo_imagenes.py --fuente imagenes/ --modelo yolo11n.pt --map
```

### Salidas
- `salida_yolo_imagenes/`: imágenes anotadas con cuadros delimitadores (*bounding boxes*), etiquetas de clase y porcentajes de confianza.
- `salida_yolo_imagenes/resultados_yolo_imagenes.csv`: registro detallado por imagen de tiempo de inferencia (ms), cantidad de objetos detectados, confianza media y clases identificadas.

### Métricas


| Métrica | Valor | Tipo de Métrica |
|---|---|---|
| Tiempo medio por imagen (ms) | 91.50 ms | Rendimiento / Velocidad |
| FPS equivalente | 10.93 | Rendimiento / Velocidad |
| Detecciones promedio | 8.62 | Comportamiento en prueba |
| Confianza media | 0.644 | Comportamiento en prueba |
| Precision | 0.663 | Calidad / Exactitud (COCO128) |
| Recall | 0.589 | Calidad / Cobertura (COCO128) |
| mAP@0.50 | 0.671 | Estándar de la industria |
| mAP@0.5:0.95 | 0.503 | Estándar riguroso |

### Observaciones
- **Hardware y rendimiento:** La inferencia se ejecutó completamente en CPU (AMD Ryzen 5 4500U with Radeon Graphics) alcanzando una tasa de procesamiento de ~10.93 FPS y una latencia media de 91.50 ms, demostrando la eficiencia del diseño de una sola etapa de YOLO en entornos sin aceleración por GPU dedicada.
- **Escenarios de alta densidad:** En imágenes complejas como tráfico vehicular y ciclistas urbanos (`images (1).jpg` e `images (5).jpg`), el detector identificó hasta 22 y 16 instancias simultáneas (`car`, `person`, `bicycle`, `truck`), manteniendo una correcta separación y discriminación espacial entre objetos superpuestos.
- **Escenas cotidianas e interiores:** Logró identificar con alta certeza mascotas (`cat`, `dog`) y elementos de oficina/hogar (`laptop`, `cup`, `chair`) promediando una confianza general de 0.644.
- **Evaluación benchmark:** Evaluado sobre el subconjunto COCO128 mediante `--map`, el modelo `yolo11n` (2.6M de parámetros y 6.5 GFLOPs) alcanzó un $mAP_{50}$ de 0.671 y un $mAP_{50-95}$ de 0.503, ofreciendo un balance óptimo entre bajo costo computacional y precisión de detección.