import cv2
from ultralytics import YOLO

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
# Modelo YOLOv8 nano (rápido, ideal para tiempo real en CPU)
modelo = YOLO('yolov8n.pt')

TITULO_VENTANA = 'Deteccion YOLO - Animales, Personas y Vehiculos'

# IDs de las clases de COCO que queremos detectar:
# 0=persona, 2=coche, 3=moto, 7=camión,
# 14=pájaro, 15=gato, 16=perro, 17=caballo, 18=oveja, 19=vaca
CLASES_INTERES = [0, 2, 3, 7, 14, 15, 16, 17, 18, 19]

# Nombres en español para mostrar en pantalla
NOMBRES_ES = {
    0: 'Persona',   2: 'Coche',     3: 'Moto',
    7: 'Camión',    14: 'Pájaro',   15: 'Gato',
    16: 'Perro',    17: 'Caballo',  18: 'Oveja',
    19: 'Vaca',
}

# Umbral de confianza (0.5 = 50%). Súbelo si hay falsos positivos.
UMBRAL_CONFIANZA = 0.45


# ============================================================
# 2. FUNCIONES
# ============================================================
def detectar_y_anotar(frame):
    """Ejecuta YOLO sobre un frame y devuelve la imagen anotada
    (cajas + panel con la lista de objetos detectados)."""

    # --- Inferencia YOLO ---
    resultados = modelo(
        frame,
        conf=UMBRAL_CONFIANZA,   # umbral de confianza
        classes=CLASES_INTERES,  # solo estas clases
        verbose=False            # no imprimir logs en consola
    )

    # --- Dibujar cajas y etiquetas ---
    # .plot() devuelve el frame con las cajas, nombres y confianza
    frame_anotado = resultados[0].plot()

    # --- Panel con la lista de objetos detectados ---
    # Contar cuántas veces aparece cada clase
    conteo = {}
    for caja in resultados[0].boxes:
        clase_id = int(caja.cls[0])
        conteo[clase_id] = conteo.get(clase_id, 0) + 1

    alto_frame, ancho_frame = frame_anotado.shape[:2]
    lineas_panel = [f"Objetos: {len(resultados[0].boxes)}"]
    for clase_id, cantidad in sorted(conteo.items(),
                                     key=lambda x: -x[1]):
        nombre = NOMBRES_ES.get(clase_id,
                                resultados[0].names.get(clase_id, '?'))
        lineas_panel.append(f"  {nombre}: {cantidad}")
    if len(lineas_panel) == 1:
        lineas_panel.append("  (ninguno)")

    # Fondo semitransparente del panel (arriba a la derecha)
    ancho_panel = 220
    alto_panel = 30 + 28 * len(lineas_panel)
    x0 = ancho_frame - ancho_panel - 10
    overlay = frame_anotado.copy()
    cv2.rectangle(overlay, (x0, 10), (ancho_frame - 10, 10 + alto_panel),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame_anotado, 0.4, 0,
                    frame_anotado)
    cv2.rectangle(frame_anotado, (x0, 10), (ancho_frame - 10, 10 + alto_panel),
                  (0, 200, 255), 2)

    # Texto del panel
    for i, texto in enumerate(lineas_panel):
        cv2.putText(frame_anotado, texto,
                    (x0 + 10, 40 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255) if i == 0 else (180, 255, 180),
                    2 if i == 0 else 1)

    return frame_anotado


def procesar_imagen(ruta):
    """Detecta objetos en una foto/imagen y la muestra en pantalla."""
    frame = cv2.imread(ruta)
    if frame is None:
        print(f"❌ Error: no se pudo abrir la imagen '{ruta}'.")
        return

    # Si la imagen es muy grande para la pantalla, redimensionar
    alto, ancho = frame.shape[:2]
    escala = min(1.0, 1280 / ancho, 720 / alto)
    if escala < 1.0:
        frame = cv2.resize(frame, (int(ancho * escala), int(alto * escala)))

    frame_anotado = detectar_y_anotar(frame)
    cv2.imshow(TITULO_VENTANA, frame_anotado)
    print("✅ Imagen procesada. Pulsa cualquier tecla para salir.")
    cv2.waitKey(0)


def procesar_video(fuente):
    """Bucle de detección para cámara (0) o archivo de video."""
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print("❌ Error: no se pudo abrir la fuente de video.")
        return

    # (Opcional) Guardar el resultado en un archivo .mp4
    guardar_video = False          # Cambia a True para grabar la salida
    if guardar_video:
        ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        salida = cv2.VideoWriter('salida_deteccion.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (ancho, alto))

    print("✅ Detección iniciada. Pulsa 'q' o ESC para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ℹ️ Fin del video.")
            break

        frame_anotado = detectar_y_anotar(frame)

        # --- Mostrar resultado ---
        cv2.imshow(TITULO_VENTANA, frame_anotado)

        # --- Guardar frame si se solicitó ---
        if guardar_video:
            salida.write(frame_anotado)

        # Salir con la tecla 'q', ESC o cerrando la ventana
        tecla = cv2.waitKey(1) & 0xFF
        ventana_cerrada = cv2.getWindowProperty(
            TITULO_VENTANA, cv2.WND_PROP_VISIBLE) < 1
        if tecla == ord('q') or tecla == 27 or ventana_cerrada:
            break

    cap.release()
    if guardar_video:
        salida.release()
    cv2.destroyAllWindows()
    print("🔚 Programa finalizado.")


# ============================================================
# 3. MENÚ DE SELECCIÓN DE FUENTE
# ============================================================
print("============================================")
print("   DETECCIÓN DE OBJETOS CON YOLO")
print("============================================")
print("  1 = Cámara web (tiempo real)")
print("  2 = Archivo de video (.mp4, .avi...)")
print("  3 = Imagen / foto (.jpg, .png...)")
print("============================================")

opcion = input("Elige una opción [1/2/3]: ").strip()

if opcion == '1':
    procesar_video(0)
elif opcion == '2':
    ruta = input("Ruta del video: ").strip().strip('"')
    if ruta:
        procesar_video(ruta)
    else:
        print("❌ No se indicó ninguna ruta.")
elif opcion == '3':
    ruta = input("Ruta de la imagen: ").strip().strip('"')
    if ruta:
        procesar_imagen(ruta)
    else:
        print("❌ No se indicó ninguna ruta.")
else:
    print("❌ Opción no válida.")
