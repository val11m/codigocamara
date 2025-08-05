from ultralytics import YOLO

import cv2

import winsound  # Importamos winsound para emitir sonido en Windows

# === Parámetros de capacidad ===

CAPACIDAD_TOTAL = 3  # Ajusta según tu necesidad

# === Cargar modelo YOLOv8 ===

model = YOLO("yolov8n.pt")

# === Iniciar captura de video desde la cámara ===

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("No se pudo acceder a la cámara.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

        # === Inferencia con YOLO ===

    results = model(frame)[0]

    # === Contar personas (clase 0) ===

    total_personas = sum(1 for cls in results.boxes.cls if int(cls) == 0)

    # === Determinar mensaje y sonido según capacidad ===

    if total_personas >= CAPACIDAD_TOTAL:

        mensaje = "BUS LLENO - Capacidad máxima alcanzada"

        color = (0, 0, 255)  # Rojo

        winsound.Beep(1000, 500)  # Frecuencia 1000Hz, duración 500ms

    elif total_personas >= CAPACIDAD_TOTAL - 10:

        mensaje = "BUS CASI LLENO - Espere otra unidad"

        color = (0, 165, 255)  # Naranja

    elif total_personas > 0:

        mensaje = f"Personas detectadas: {total_personas} - Espacio disponible"

        color = (0, 255, 0)  # Verde

    else:

        mensaje = "BUS VACÍO"

        color = (255, 255, 0)  # Amarillo

    # === Dibujar cajas para personas ===

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):

        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # === Mostrar mensaje en la imagen ===

    cv2.putText(

        frame,

        mensaje,

        (10, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        color,

        2

    )

    # === Mostrar imagen en ventana ===

    cv2.imshow("Detección de personas - Bus", frame)

    # === Salir con 'q' ===

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # === Liberar cámara y cerrar ventanas ===

cap.release()

cv2.destroyAllWindows()