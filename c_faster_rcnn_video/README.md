# Actividad 06 — Ítem C: Detección de objetos en video con Faster R-CNN

**Curso:** Visión Artificial — Segunda Unidad  
**Responsable:** Jhoel Yovani Ticona Erquinigo  
**Universidad:** Universidad Nacional del Altiplano — Puno  
**Fecha de presentación:** 09/11/2026

## 1. Descripción

Aplicación de detección de objetos en video y webcam utilizando **Faster R-CNN MobileNetV3 Large FPN 320**, con pesos preentrenados en el conjunto de datos COCO.

El proyecto utiliza Python, PyTorch, Torchvision y OpenCV para identificar objetos, mostrar cuadros delimitadores (*bounding boxes*) y evaluar el rendimiento del modelo durante el procesamiento de video.

## 2. Tecnologías utilizadas

- Python 3.10 o superior
- PyTorch y Torchvision
- OpenCV
- NumPy
- CUDA (opcional para GPU NVIDIA)

## 3. Instalación

Abrir una terminal en la carpeta `c_faster_rcnn_video`.

**Crear entorno virtual:**

```cmd
python -m venv .venv
```

**Activar entorno virtual en CMD:**

```cmd
.venv\Scripts\activate.bat
```

En PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

**Instalar PyTorch con CUDA 11.8:**

```cmd
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Instalar dependencias adicionales:**

```cmd
pip install opencv-python numpy
```

Para utilizar exclusivamente CPU:

```cmd
pip install torch torchvision opencv-python numpy
```

La instalación de PyTorch y Torchvision debe ser compatible con el hardware y la versión de Python utilizada. Consultar la [documentación oficial de PyTorch](https://pytorch.org/get-started/locally/).

## 4. Ejecución

**Detección con webcam:**

```cmd
python detector_faster_rcnn.py
```

**Detección mediante video local:**

```cmd
python detector_faster_rcnn.py --video prueba.mp4
```

**Detección con almacenamiento del video:**

```cmd
python detector_faster_rcnn.py --guardar
```

**Modificar umbral de confianza:**

```cmd
python detector_faster_rcnn.py --conf 0.5
```

**Modificar ancho del fotograma:**

```cmd
python detector_faster_rcnn.py --width 640
```

### Controles

- `Q`: finalizar la ejecución.
- `S`: guardar una captura de pantalla.

La primera ejecución puede descargar automáticamente los pesos preentrenados del modelo.

## 5. Funcionamiento

El procesamiento se realiza mediante las siguientes etapas:

1. **Captura:** OpenCV obtiene los fotogramas de la webcam o video.
2. **Preprocesamiento:** se acondicionan los fotogramas para el modelo.
3. **Propuesta de regiones:** Faster R-CNN identifica regiones candidatas mediante su RPN (*Region Proposal Network*).
4. **Clasificación y localización:** se predicen las clases y coordenadas de los objetos.
5. **Visualización:** se dibujan los cuadros delimitadores y las etiquetas.
6. **Medición:** se registran las métricas de rendimiento.

## 6. Métricas y evidencias

Los resultados se almacenan en la carpeta `resultados/`.

| Archivo | Descripción |
|---|---|
| `metricas.csv` | Latencia, FPS y número de detecciones por fotograma |
| `captura_*.jpg` | Capturas de las detecciones |
| `deteccion.mp4` | Video procesado, cuando se utiliza `--guardar` |

**Métricas:**

- **FPS:** cantidad de fotogramas procesados por segundo.
- **Latencia (ms):** tiempo requerido por el modelo para procesar un fotograma.
- **Número de detecciones:** cantidad de objetos identificados por fotograma.
- **Confianza:** puntuación asignada por el modelo a cada detección.

La latencia de inferencia y los FPS del procesamiento completo son medidas distintas, ya que los FPS incluyen operaciones adicionales como captura y visualización.

## 7. Modelo utilizado

Se utiliza **Faster R-CNN MobileNetV3 Large FPN 320**, disponible en Torchvision y preentrenado con COCO.

Faster R-CNN es un detector de dos etapas:

1. Una red de propuestas de regiones identifica ubicaciones candidatas.
2. Una segunda etapa clasifica los objetos y ajusta sus cuadros delimitadores.

El modelo permite detectar distintas categorías de objetos comunes. Su rendimiento depende del hardware, la resolución y la complejidad de la escena.

## 8. Comparación con YOLO

La comparación con YOLO se realizará utilizando condiciones experimentales equivalentes, procurando mantener el mismo video, resolución y dispositivo.

Se analizarán:

- FPS promedio.
- Latencia promedio de inferencia.
- Cantidad de objetos detectados.
- Comportamiento ante movimiento.
- Diferencias de velocidad y estabilidad.

Para evaluar precisión mediante mAP se requieren imágenes o videos anotados con cuadros delimitadores de referencia.

## 9. Resultados experimentales

*Pendiente de completar con las mediciones obtenidas durante las pruebas.*

| Métrica | Faster R-CNN |
|---|---|
| FPS promedio | Pendiente |
| Latencia promedio | Pendiente |
| Dispositivo utilizado | Pendiente |
| Resolución evaluada | Pendiente |

## 10. Conclusiones

Las conclusiones se establecerán a partir de los resultados experimentales, considerando la capacidad del modelo para detectar objetos en video, su velocidad de procesamiento y su comportamiento frente a YOLO.