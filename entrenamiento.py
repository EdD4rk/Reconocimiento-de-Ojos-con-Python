# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import random
from PIL import Image

# Usamos la misma arquitectura que en el script de reconocimiento
from reconocimiento_ojos import EyeEncoder

# ==============================================================================
# 1. DATASET PERSONALIZADO PARA TRIPLETS
# ==============================================================================
class TripletDataset(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
        # Crear un diccionario para encontrar fácilmente las imágenes por clase
        self.labels_to_indices = {label: np.where(np.array(self.image_folder_dataset.targets) == label)[0]
                                  for label in set(self.image_folder_dataset.targets)}

    def __getitem__(self, index):
        # Anchor
        img1_path, label1 = self.image_folder_dataset.samples[index]
        
        # Positive: elegir otra imagen de la misma clase
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.labels_to_indices[label1])
        img2_path, _ = self.image_folder_dataset.samples[positive_index]
        
        # Negative: elegir una imagen de una clase diferente
        label2 = random.choice(list(filter(lambda x: x != label1, self.labels_to_indices.keys())))
        negative_index = random.choice(self.labels_to_indices[label2])
        img3_path, _ = self.image_folder_dataset.samples[negative_index]

        # Cargar imágenes
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img3 = Image.open(img3_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.image_folder_dataset)

# ==============================================================================
# 2. BUCLE DE ENTRENAMIENTO
# ==============================================================================
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # --- Cargar y transformar datos ---
    data_dir = 'dataset_identities'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        full_dataset = ImageFolder(root=data_dir)
        triplet_dataset = TripletDataset(image_folder_dataset=full_dataset, transform=transform)
        train_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True, num_workers=4)
        print(f"Dataset cargado. Encontradas {len(full_dataset.classes)} clases (personas).")
    except FileNotFoundError:
        print(f"Error: El directorio '{data_dir}' no fue encontrado.")
        print("Por favor, crea el dataset con la estructura de carpetas correcta.")
        return

    # --- Modelo, Loss y Optimizador ---
    model = EyeEncoder().to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # --- Bucle de entrenamiento ---
    num_epochs = 20 # Puedes aumentar este número para mejores resultados
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            
            output_anchor = model(anchor)
            output_positive = model(positive)
            output_negative = model(negative)
            
            loss = criterion(output_anchor, output_positive, output_negative)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9: # Imprimir cada 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Entrenamiento finalizado.')
    
    # --- Guardar el modelo entrenado ---
    torch.save(model.state_dict(), 'modelo_ojos.pth')
    print("Modelo guardado como 'modelo_ojos.pth'")

if __name__ == '__main__':
    train()