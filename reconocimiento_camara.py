# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import pyttsx3 # <--- Librería para la alerta de voz
import time    # <--- Para controlar la frecuencia de las alertas

# ==============================================================================
# (Esta parte es idéntica a los otros scripts)
# 1. ARQUITECTURA DEL MODELO (ENCODER)
# ==============================================================================
class EyeEncoder(nn.Module):
    def __init__(self):
        super(EyeEncoder, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 29 * 29, 256), nn.PReLU(),
            nn.Linear(256, 256), nn.PReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

# ==============================================================================
# 2. FUNCIÓN DE DETECCIÓN DE OJOS (usando OpenCV)
# ==============================================================================
def detectar_ojos(frame: np.ndarray) -> np.ndarray:
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ojos = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return np.array([[x, y, x + w, y + h] for (x, y, w, h) in ojos]).astype(int)

# ==============================================================================
# 3. PIPELINE DE RECONOCIMIENTO EN TIEMPO REAL
# ==============================================================================
def main():
    # --- CONFIGURACIÓN INICIAL ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # --- INICIALIZAR ALERTA DE VOZ ---
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # de acuerdo al numero, ralentiza la voz, ponlo más bajo para hacerlo más lento
    last_seen = {}  # Diccionario para no repetir la alerta constantemente
    ALERT_COOLDOWN = 5  # Segundos de espera antes de repetir una alerta para la misma persona

    # --- CARGAR EL MODELO ENTRENADO ---
    try:
        encoder = EyeEncoder().to(device)
        encoder.load_state_dict(torch.load('modelo_ojos.pth', map_location=device))
        encoder.eval()
        print("Modelo 'modelo_ojos.pth' cargado correctamente.")
    except FileNotFoundError:
        print("Error: No se encontró 'modelo_ojos.pth'. Asegúrate de que esté en la misma carpeta.")
        return

    # --- BASE DE DATOS DE REFERENCIA (SIMULACIÓN) ---
    print("Creando base de datos de referencia...")
    db_embeddings = {}
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Aquí deberías tener una imagen de referencia por cada persona entrenada
    referencias = {
        "Persona 1": "ejemplo_persona_a.jpg",  # Cambia "Persona_A" por el nombre real de la carpeta
        "Persona 2": "ejemplo_persona_b.jpg"   # Cambia "Persona_B" por el nombre real de la carpeta
    }

    # Crear embeddings de referencia
    for nombre, ruta_img in referencias.items():
        try:
            img_ref = cv2.imread(ruta_img)
            ojos_ref = detectar_ojos(img_ref)
            if len(ojos_ref) > 0:
                x1, y1, x2, y2 = ojos_ref[0]
                ojo_recortado = img_ref[y1:y2, x1:x2]
                ojo_tensor = transform(Image.fromarray(cv2.cvtColor(ojo_recortado, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                with torch.no_grad():
                    db_embeddings[nombre] = encoder(ojo_tensor)
                print(f"Embedding de '{nombre}' creado y guardado.")
        except Exception as e:
            print(f"No se pudo procesar la referencia para '{nombre}' en la ruta '{ruta_img}'. Error: {e}")

    if not db_embeddings:
        print("La base de datos de embeddings está vacía. No se puede continuar.")
        return

    # --- INICIAR CÁMARA ---
    cap = cv2.VideoCapture(0)  # 0 es usualmente la cámara integrada
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("\nIniciando reconocimiento en tiempo real. Presiona 'q' para salir.")

    # --- BUCLE PRINCIPAL (CICLO EN TIEMPO REAL) ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detectar todos los ojos en el fotograma actual
        bboxes = detectar_ojos(frame)

        # Ordenar ojos de izquierda a derecha para etiquetarlos correctamente
        bboxes = sorted(bboxes, key=lambda b: b[0])

        # 2. Para cada ojo detectado...
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            
            # Recortar el ojo
            ojo_recortado = frame[y1:y2, x1:x2]

            # Preprocesar para el modelo
            ojo_tensor = transform(Image.fromarray(cv2.cvtColor(ojo_recortado, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            
            # Generar su embedding
            with torch.no_grad():
                embedding_actual = encoder(ojo_tensor)

            # Comparar con la base de datos
            identidad = 'Persona'
            dist_min = float('inf')
            
            for nombre, embedding_ref in db_embeddings.items():
                dist = torch.nn.functional.pairwise_distance(embedding_actual, embedding_ref)
                if dist < dist_min:
                    dist_min = dist
                    if dist < 0.7:  # Umbral de decisión (ajustable)
                        identidad = nombre
            
            # --- Poner etiquetas fijas según posición del ojo ---
            etiqueta = "Ojo derecho" if i == 0 else "Ojo izquierdo"
            color = (0, 255, 0)
            
            # Dibujar rectángulo enmarcando el ojo
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Escribir la etiqueta en vez del número
            cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Alerta de voz si se reconoce a alguien
            if identidad != 'Ojo':
                current_time = time.time()
                # Si la persona no ha sido vista o ha pasado el tiempo de cooldown
                if identidad not in last_seen or (current_time - last_seen[identidad]) > ALERT_COOLDOWN:
                    mensaje = f"Ojo de {identidad} detectado"
                    engine.say(mensaje)
                    engine.runAndWait()
                    last_seen[identidad] = current_time  # Actualizar el tiempo de la última vista

        # 3. Mostrar el fotograma resultante
        cv2.imshow('Reconocimiento de Ojos en Tiempo Real', frame)

        # 4. Condición de salida (presionar 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- LIMPIEZA ---
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara liberada y ventanas cerradas.")


if __name__ == '__main__':
    main()