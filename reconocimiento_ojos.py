# -*- coding: utf-8 -*-
################################################################################
#                   RECONOCIMIENTO DE OJOS CON DEEP LEARNING                   #
#                                                                              #
# Adaptado del trabajo original de Joaquín Amat Rodrigo (cienciadedatos.net)   #
################################################################################

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ==============================================================================
# 1. ARQUITECTURA DEL MODELO (ENCODER)
# Esta es la red neuronal que convertirá una imagen de un ojo en un vector
# numérico (embedding). DEBE ser la misma arquitectura que uses para entrenar.
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
            nn.Linear(256, 128) # El embedding final será de 128 dimensiones
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

# ==============================================================================
# 2. FUNCIONES DE DETECCIÓN Y PROCESAMIENTO
# Aquí reemplazamos MTCNN por el detector de ojos de OpenCV.
# ==============================================================================
def detectar_ojos(imagen: np.ndarray) -> np.ndarray:
    """
    Detecta la posición de ojos en una imagen empleando un clasificador de Haar.
    Devuelve las bounding boxes en formato [x1, y1, x2, y2].
    """
    # Cargar el clasificador pre-entrenado de OpenCV para ojos
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar los ojos
    ojos = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Convertir el formato de [x, y, w, h] a [x1, y1, x2, y2]
    bboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in ojos])
    
    return bboxes.astype(int)

def mostrar_bboxes(imagen: np.ndarray, bboxes: np.ndarray, identidades: list = None, ax=None):
    """
    Muestra la imagen original con las bounding box de los ojos detectados.
    """
    if ax is None:
        ax = plt.gca()
        
    ax.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    
    if identidades is None:
        identidades = [None] * len(bboxes)
        
    for i, bbox in enumerate(bboxes):
        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(rect)
        if identidades[i] is not None:
            ax.text(
                x=bbox[0],
                y=bbox[1] - 5,
                s=identidades[i],
                color='white',
                backgroundcolor='green'
            )

# ==============================================================================
# 3. PIPELINE DE RECONOCIMIENTO
# ==============================================================================
def main():
    # --- CONFIGURACIÓN ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    
    # Cargar el modelo encoder entrenado
    try:
        encoder = EyeEncoder().to(device)
        encoder.load_state_dict(torch.load('modelo_ojos.pth', map_location=device))
        encoder.eval()
        print("Modelo 'modelo_ojos.pth' cargado correctamente.")
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'modelo_ojos.pth'.")
        print("Por favor, entrena el modelo usando la guía y el script de entrenamiento.")
        return

    # --- BASE DE DATOS DE REFERENCIA ---
    # En un caso real, cargarías esto desde un archivo.
    # Aquí creamos una base de datos de ejemplo.
    # 1. Carga las imágenes de las personas que conoces.
    # 2. Detecta sus ojos.
    # 3. Genera y guarda sus embeddings.
    
    print("Creando base de datos de referencia (simulación)...")
    db_embeddings = {}
    
    # --- EJEMPLO CON UNA PERSONA (DEBES SUSTITUIR ESTO) ---
    # Asume que tienes una foto clara del ojo de "Persona A"
    try:
        img_persona_a = cv2.imread('ejemplo_persona_a.jpg') # ¡Crea esta imagen!
        ojos_a = detectar_ojos(img_persona_a)
        
        if len(ojos_a) > 0:
            # Usamos el primer ojo detectado como referencia
            x1, y1, x2, y2 = ojos_a[0]
            ojo_recortado_a = img_persona_a[y1:y2, x1:x2]
            
            # Preprocesar la imagen del ojo para el modelo
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            ojo_tensor_a = transform(Image.fromarray(cv2.cvtColor(ojo_recortado_a, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            
            # Generar y guardar el embedding
            with torch.no_grad():
                embedding_a = encoder(ojo_tensor_a)
                db_embeddings['Persona_A'] = embedding_a
            print("Embedding de 'Persona_A' guardado.")

    except Exception as e:
        print("No se pudo crear la referencia para Persona_A. Asegúrate de tener 'ejemplo_persona_a.jpg'")
        print(f"Error: {e}")

    # --- RECONOCIMIENTO EN UNA NUEVA IMAGEN ---
    try:
        img_test = cv2.imread('imagen_test.jpg') # ¡Crea esta imagen con una o más personas!
        ojos_test = detectar_ojos(img_test)
        
        identidades = []
        
        for bbox in ojos_test:
            x1, y1, x2, y2 = bbox
            ojo_recortado = img_test[y1:y2, x1:x2]
            
            # Preprocesar
            ojo_tensor = transform(Image.fromarray(cv2.cvtColor(ojo_recortado, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            
            # Generar embedding del ojo a identificar
            with torch.no_grad():
                embedding_actual = encoder(ojo_tensor)
            
            # Comparar con la base de datos
            identidad_encontrada = 'Desconocido'
            distancia_minima = float('inf')
            
            for nombre, embedding_ref in db_embeddings.items():
                distancia = torch.nn.functional.pairwise_distance(embedding_actual, embedding_ref)
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    # Umbral de decisión: si la distancia es baja, es la misma persona
                    if distancia < 0.7: # Puedes ajustar este umbral
                        identidad_encontrada = nombre
            
            identidades.append(f'{identidad_encontrada} ({distancia_minima.item():.2f})')
        
        # Mostrar resultado
        fig, ax = plt.subplots(figsize=(10, 8))
        mostrar_bboxes(img_test, ojos_test, identidades, ax)
        plt.show()

    except Exception as e:
        print("No se pudo procesar 'imagen_test.jpg'. Asegúrate de que el archivo existe.")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()