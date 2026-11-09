
# ACTIVIDAD 06 – COMPARACIÓN DE ALGORITMOS DE DETECCIÓN DE OBJETOS: FASTER R-CNN Y YOLO

## 1. INFORMACIÓN GENERAL

**Asignatura:** Visión Artificial  
**Actividad:** Actividad 06 – Segunda unidad  
**Tema:** Comparación de algoritmos de detección de objetos mediante Faster R-CNN y YOLO.  

### Integrantes

1. Willebeldo Fermin. Quispe Soncco
2. Alex C. Mestas Calsin
3. José M. Ancco García
4. Benigno S. Mescco Condori
5. Jhoel Y. Ticona Erquinigo

---

## 2. INTRODUCCIÓN

La detección de objetos es una de las aplicaciones más importantes de la visión artificial y el aprendizaje profundo. Esta tecnología permite identificar y localizar objetos dentro de imágenes y videos mediante modelos entrenados con grandes conjuntos de datos.

En esta actividad se desarrolla una comparación entre dos enfoques de detección de objetos: Faster R-CNN y YOLO. Para ello, se implementan aplicaciones que permiten procesar imágenes y videos, analizar los objetos detectados y obtener métricas relacionadas con el rendimiento de cada modelo.

Faster R-CNN utiliza una arquitectura basada en regiones para localizar y clasificar objetos, mientras que YOLO realiza la detección mediante un enfoque de una sola etapa, buscando ofrecer una respuesta rápida durante la inferencia.

El propósito es analizar el comportamiento de ambos enfoques, considerando aspectos como precisión, tiempo de procesamiento, velocidad de detección y cantidad de objetos identificados.

---

## 3. OBJETIVOS

### 3.1. Objetivo general

Implementar y comparar algoritmos de detección de objetos basados en Faster R-CNN y YOLO, utilizando imágenes y videos para analizar su rendimiento mediante diferentes métricas de evaluación.

### 3.2. Objetivos específicos

- Implementar Faster R-CNN para detectar objetos en imágenes.
- Implementar YOLO para detectar objetos en imágenes.
- Implementar Faster R-CNN para detectar objetos en videos o mediante una cámara en tiempo real.
- Implementar YOLO para detectar objetos en videos o mediante una cámara en tiempo real.
- Comparar el rendimiento de los algoritmos mediante métricas de precisión, velocidad y tiempo de procesamiento.

---

## 4. TECNOLOGÍAS Y HERRAMIENTAS UTILIZADAS

| Herramienta | Descripción |
|---|---|
| Python | Lenguaje de programación utilizado. |
| OpenCV | Procesamiento de imágenes y captura de video. |
| PyTorch | Ejecución del modelo Faster R-CNN. |
| Torchvision | Modelos y herramientas de visión artificial. |
| Ultralytics | Implementación del modelo YOLO. |
| ONNX Runtime | Ejecución del modelo Faster R-CNN mediante ONNX. |
| NumPy | Operaciones numéricas. |
| Pandas | Organización y almacenamiento de métricas. |
| Matplotlib | Generación de gráficos de resultados. |

---

## 5. DESARROLLO DE LAS APLICACIONES

Se han desarrollado tres de las cuatro aplicaciones solicitadas en la actividad. Cada implementación utiliza modelos preentrenados para realizar la detección de objetos.

### 5.1. Aplicación A: Faster R-CNN en imágenes

**Archivo:** `a_faster_rcnn_imagenes.py`

Esta aplicación implementa el algoritmo Faster R-CNN utilizando la arquitectura ResNet-50 con Feature Pyramid Network (FPN), entrenada previamente con el conjunto de datos COCO.

El programa permite detectar y clasificar objetos presentes en imágenes, dibujando cuadros delimitadores alrededor de cada objeto identificado.

También permite utilizar dos alternativas para ejecutar el modelo:

- ONNX Runtime.
- Torchvision con PyTorch.

La aplicación contempla la ejecución mediante CPU o GPU CUDA, dependiendo de la disponibilidad del hardware.

**Características principales:**

- Detección de objetos en imágenes.
- Utilización del modelo Faster R-CNN ResNet-50 FPN.
- Reconocimiento de las 80 categorías del conjunto de datos COCO.
- Generación de imágenes con las detecciones dibujadas.
- Cálculo del tiempo medio, mediana y percentil 95 de latencia.
- Cálculo de FPS equivalentes.
- Evaluación de mAP, precisión, recall y F1 cuando se proporcionan etiquetas de referencia.
- Generación de archivos JSON y gráficos con los resultados.

**Métricas consideradas:**

- mAP@[0.5:0.95].
- mAP@0.5.
- mAP@0.75.
- AP para objetos pequeños, medianos y grandes.
- AR@100.
- Precisión.
- Recall.
- F1-score.
- Latencia y FPS.

**Ejecución:**

```bash
python a_faster_rcnn_imagenes.py
```

Para utilizar imágenes propias, se debe especificar la carpeta correspondiente mediante los argumentos admitidos por el programa.

**Importante:** cuando se utilizan imágenes personalizadas, es necesario proporcionar sus etiquetas correspondientes para calcular las métricas de evaluación como mAP. Sin etiquetas de referencia, el programa permite medir principalmente el rendimiento de inferencia.

### 5.2. Aplicación B: YOLO en imágenes

**Archivo:** `yolo_imagenes.py`

Esta aplicación utiliza el modelo YOLO11n mediante la biblioteca Ultralytics para realizar detecciones de objetos en imágenes.

El programa recibe una carpeta de imágenes, ejecuta el modelo y genera nuevas imágenes con los objetos detectados, sus cuadros delimitadores y sus respectivas etiquetas.

Además, almacena información sobre el rendimiento de cada imagen procesada en un archivo CSV.

**Características principales:**

- Detección de objetos en imágenes.
- Utilización del modelo `yolo11n.pt`.
- Procesamiento de diferentes formatos de imagen.
- Generación de imágenes con objetos detectados.
- Registro del tiempo de inferencia.
- Cálculo de la cantidad de detecciones.
- Cálculo de la confianza promedio.
- Almacenamiento de resultados en formato CSV.
- Evaluación opcional utilizando el conjunto de datos COCO128.

**Métricas consideradas:**

- Tiempo de inferencia por imagen.
- FPS equivalentes.
- Cantidad de objetos detectados.
- Confianza promedio.
- Precision.
- Recall.
- mAP@0.50.
- mAP@0.50:0.95.

**Instalación de dependencias:**

```bash
pip install ultralytics opencv-python pandas
```

**Ejecución:**

```bash
python yolo_imagenes.py --fuente imagenes/ --modelo yolo11n.pt --map
```

**Archivos generados:**

- `salida_yolo_imagenes/`: imágenes procesadas.
- `resultados_yolo_imagenes.csv`: métricas obtenidas durante la detección.

**Nota sobre la evaluación:** la opción `--map` ejecuta la evaluación utilizando COCO128. Para realizar una comparación directa de mAP con Faster R-CNN, ambos modelos deben evaluarse sobre el mismo conjunto de datos y bajo condiciones equivalentes.

### 5.3. Aplicación C: Faster R-CNN en videos y tiempo real

Esta aplicación implementa Faster R-CNN para detectar objetos en secuencias de video o mediante una cámara web.

A diferencia de la aplicación anterior, esta implementación utiliza la arquitectura Faster R-CNN MobileNet V3 Large 320 FPN, con pesos preentrenados en COCO.

El programa captura los fotogramas del video, los procesa individualmente y muestra las detecciones realizadas en tiempo real.

**Características principales:**

- Detección de objetos en videos.
- Captura de imágenes mediante cámara web.
- Utilización del modelo Faster R-CNN MobileNet V3 Large 320 FPN.
- Procesamiento de fotogramas individuales.
- Visualización de cuadros delimitadores y etiquetas.
- Configuración del umbral de confianza.
- Ejecución mediante CPU o GPU CUDA.
- Visualización del tiempo de procesamiento y FPS.
- Almacenamiento opcional del video procesado.
- Registro de métricas en un archivo CSV.

**Métricas consideradas:**

- Latencia por fotograma.
- FPS de procesamiento.
- Cantidad de objetos detectados por fotograma.

**Ejecución mediante cámara web:**

```bash
python c_faster_rcnn_video.py
```

**Ejecución con un video:**

```bash
python c_faster_rcnn_video.py --video videos/entrada.mp4
```

**Guardar el video procesado:**

```bash
python c_faster_rcnn_video.py --guardar
```

**Controles del programa:**

- `Q`: finalizar la ejecución.
- `S`: guardar una captura de pantalla.

**Archivos generados:**

- `resultados/deteccion.mp4`: video procesado, cuando se activa la opción de guardado.
- `resultados/metricas.csv`: métricas registradas durante el procesamiento.

**Observación:** esta aplicación utiliza MobileNet V3 Large 320 FPN, mientras que la aplicación A utiliza ResNet-50 FPN. Por lo tanto, aunque ambas pertenecen a la familia Faster R-CNN, sus resultados no representan una comparación entre arquitecturas idénticas.

### 5.4. Aplicación D: YOLO en videos y tiempo real

**Estado: Pendiente de implementación.**

Esta aplicación está contemplada en la actividad, pero todavía no se cuenta con el código correspondiente.

Su objetivo será utilizar YOLO para detectar objetos en videos o mediante una cámara web, mostrando las detecciones en tiempo real.

Se deberán considerar métricas equivalentes a las utilizadas en la aplicación C:

- Latencia por fotograma.
- FPS de procesamiento.
- Cantidad de objetos detectados.
- Confianza de las detecciones.

Una vez implementada, se incorporarán las instrucciones de ejecución, características técnicas, capturas y resultados obtenidos.

---

## 6. ESTRUCTURA DEL PROYECTO

La organización propuesta para el repositorio es la siguiente:

```text
ACTIVIDAD-06-DETECCION-OBJETOS/
│
├── README.md
├── requirements.txt
│
├── faster_rcnn/
│   ├── a_faster_rcnn_imagenes.py
│   └── c_faster_rcnn_video.py
│
├── yolo/
│   ├── b_yolo_imagenes.py
│   └── d_yolo_video.py
│
├── imagenes/
│   ├── entrada/
│   └── resultados/
│
├── videos/
│   ├── entrada/
│   └── resultados/
│
└── resultados/
    ├── metricas/
    ├── graficos/
    └── capturas/
```

**Nota:** esta estructura es una propuesta de organización. Los nombres y ubicaciones deberán ajustarse a los archivos y carpetas definitivos del proyecto.

---

## 7. INSTALACIÓN Y CONFIGURACIÓN

### 7.1. Requisitos previos

Para ejecutar las aplicaciones se recomienda contar con:

- Python 3.10 o superior.
- pip actualizado.
- Conexión a Internet para descargar los modelos preentrenados.
- Cámara web para las pruebas en tiempo real.
- GPU NVIDIA compatible con CUDA, opcional.

### 7.2. Instalación de dependencias

Se pueden instalar las principales bibliotecas utilizadas mediante:

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install pandas numpy matplotlib
pip install onnxruntime
```

Para utilizar GPU mediante ONNX Runtime, se deberá instalar la variante compatible con CUDA y configurar correctamente el entorno.

### 7.3. Ejecución

Cada aplicación debe ejecutarse desde la terminal, ubicándose previamente en la carpeta del proyecto.

Los modelos preentrenados pueden descargarse automáticamente durante la primera ejecución, dependiendo de la configuración de cada programa.

---

## 8. COMPARACIÓN DE LOS ALGORITMOS

La comparación se realizará considerando las características técnicas y las métricas generadas por cada aplicación.

### 8.1. Comparación de características

| Característica | Faster R-CNN | YOLO |
|---|---|---|
| Enfoque | Dos etapas | Una etapa |
| Detección en imágenes | Implementada | Implementada |
| Detección en videos | Implementada | Pendiente |
| Detección en tiempo real | Implementada mediante cámara web | Pendiente |
| Localización de objetos | Cuadros delimitadores | Cuadros delimitadores |
| Clasificación de objetos | Sí | Sí |
| Métricas de rendimiento | Sí | Sí |

### 8.2. Comparación de métricas

Los resultados numéricos se incorporarán después de ejecutar las aplicaciones con los conjuntos de datos y configuraciones correspondientes.

| Métrica | Faster R-CNN en imágenes | YOLO en imágenes | Faster R-CNN en video | YOLO en video |
|---|---|---|---|---|
| Precisión | Pendiente | Pendiente | No calculada | Pendiente |
| Recall | Pendiente | Pendiente | No calculado | Pendiente |
| mAP@0.50 | Pendiente | Pendiente | No calculado | Pendiente |
| mAP@0.50:0.95 | Pendiente | Pendiente | No calculado | Pendiente |
| Latencia (ms) | Pendiente | Pendiente | Pendiente | Pendiente |
| FPS | Pendiente | Pendiente | Pendiente | Pendiente |
| Objetos detectados | Pendiente | Pendiente | Pendiente | Pendiente |

**Consideraciones para una comparación justa:**

- Utilizar las mismas imágenes o videos para los modelos que se desean comparar.
- Mantener una resolución de entrada equivalente.
- Utilizar el mismo hardware cuando sea posible.
- Mantener umbrales de confianza comparables.
- Utilizar las mismas etiquetas de referencia para calcular las métricas de evaluación.
- Diferenciar los resultados de precisión de los resultados de velocidad.

Las métricas de precisión y recall requieren etiquetas de referencia. Por ello, no deben interpretarse como métricas calculadas únicamente a partir de las detecciones realizadas sobre un video sin anotaciones.

---

## 9. RESULTADOS OBTENIDOS

En esta sección se incorporarán las evidencias de las pruebas realizadas con las tres aplicaciones implementadas.

### 9.1. Resultados de Faster R-CNN en imágenes

Se agregarán las capturas de las imágenes originales y sus correspondientes resultados de detección.

**Evidencias:**

- Imagen de entrada.
- Imagen con objetos detectados.
- Archivo JSON con métricas.
- Gráficos de evaluación.

### 9.2. Resultados de YOLO en imágenes

Se agregarán las capturas de las imágenes procesadas por YOLO11n.

**Evidencias:**

- Imagen de entrada.
- Imagen con objetos detectados.
- Archivo CSV con métricas.
- Resultados de evaluación COCO128.

### 9.3. Resultados de Faster R-CNN en video

Se incorporarán capturas de la detección de objetos durante la ejecución de un video o cámara web.

**Evidencias:**

- Captura de la cámara web.
- Detecciones realizadas en los fotogramas.
- Archivo CSV con métricas.
- Video procesado, si se habilita su guardado.

### 9.4. Resultados de YOLO en video

Pendiente de implementación.

---

## 10. ANÁLISIS Y DISCUSIÓN

Las implementaciones desarrolladas permiten analizar dos enfoques diferentes para la detección de objetos.

Faster R-CNN utiliza un procedimiento basado en regiones de interés, mientras que YOLO realiza la detección mediante un enfoque de una sola etapa.

En las pruebas con imágenes, ambos modelos permiten localizar y clasificar objetos. Sin embargo, para determinar cuál presenta un mejor rendimiento es necesario analizar los resultados cuantitativos obtenidos durante las ejecuciones.

En el caso de los videos, Faster R-CNN permite registrar la latencia y los FPS durante el procesamiento de fotogramas. La comparación con YOLO en video se completará cuando se encuentre disponible la cuarta aplicación.

También se debe considerar que las dos implementaciones de Faster R-CNN utilizan arquitecturas diferentes: ResNet-50 FPN para imágenes y MobileNet V3 Large 320 FPN para video. Esta diferencia debe tenerse presente al interpretar los resultados.

---

## 11. CONCLUSIONES

1. Se implementó Faster R-CNN para la detección de objetos en imágenes mediante una arquitectura ResNet-50 FPN con pesos preentrenados en COCO.

2. Se desarrolló una aplicación basada en YOLO11n que permite detectar objetos en imágenes y almacenar métricas de rendimiento en archivos CSV.

3. Se implementó Faster R-CNN MobileNet V3 Large 320 FPN para procesar videos y realizar detecciones mediante una cámara web.

4. Las aplicaciones permiten recopilar métricas como latencia, FPS, cantidad de objetos detectados y, cuando se dispone de etiquetas de referencia, métricas de evaluación como precisión, recall y mAP.

5. La comparación cuantitativa definitiva entre Faster R-CNN y YOLO en imágenes y videos se completará cuando se ejecute la cuarta aplicación y se recopilen los resultados experimentales bajo condiciones equivalentes.

---

## 12. REFERENCIAS TECNOLÓGICAS

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. Advances in Neural Information Processing Systems.
- Ultralytics. (s. f.). *Ultralytics YOLO Documentation*. https://docs.ultralytics.com/
- PyTorch. (s. f.). *PyTorch Documentation*. https://pytorch.org/docs/stable/index.html
- Torchvision. (s. f.). *Torchvision Models Documentation*. https://docs.pytorch.org/vision/stable/models.html
- OpenCV. (s. f.). *OpenCV Documentation*. https://docs.opencv.org/

---

## 13. ESTADO ACTUAL DE LA ACTIVIDAD

| Aplicación | Descripción | Estado |
|---|---|---|
| A | Faster R-CNN en imágenes | Implementada |
| B | YOLO en imágenes | Implementada |
| C | Faster R-CNN en videos | Implementada |
| D | YOLO en videos | Pendiente |

---

**Proyecto desarrollado como parte de la Actividad 06 – Segunda unidad, correspondiente al estudio y comparación de algoritmos de detección de objetos mediante técnicas de aprendizaje profundo.**


