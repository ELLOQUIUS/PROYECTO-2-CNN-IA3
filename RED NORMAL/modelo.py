import torch
import torch.nn as nn
import torch.optim as optim

'''
Este módulo contiene funciones de activación y pérdida que no requieren ser
instanciadas como objetos. Se usa cuando quieres aplicar funciones
directamente en el forward. Ejemplos comunes:

F.relu(x)                # Activación ReLU
F.cross_entropy(logits, labels)  # Pérdida para clasificación multiclase
F.sigmoid(x)             # Activación sigmoide
F.softmax(x, dim=1)      # Softmax sobre la dimensión de clases

Útil cuando no quieres definir nn.ReLU() como capa, sino aplicar la función
directamente.
'''
import torch.nn.functional as F

'''
DataLoader es el motor de batching de PyTorch. Permite cargar datos en
mini-lotes, barajarlos y paralelizar la lectura. Ejemplo típico:

train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
'''
from torch.utils.data import DataLoader

'''
Este import trae dos cosas clave para visión por computadora:

datasets: Conjuntos de datos predefinidos como MNIST, CIFAR10, ImageNet.
datasets.MNIST(root='./data', train=True, download=True, transform=...)

transforms: Transformaciones aplicadas a las imágenes: normalización,
redimensionamiento, conversión a tensor, etc.
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

Se usan para preparar los datos antes de entrenar
'''
from torchvision import datasets, transforms

'''
Junta varias imágenes en una sola cuadrícula para visualización
(por ejemplo, en TensorBoard o matplotlib). Ejemplo de uso:

grid = make_grid(batch_of_images)

Muy útil para inspeccionar cómo se ven los datos o los resultados del modelo.
'''
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Convertir archivos de imagenes de MNIST a un tensor de 4 dimensiones 
# (número de imágenes, altura, ancho, color)
transform = transforms.ToTensor()   

# Importamos el conjunto de datos de entrenamiento de MNIST
train_data = datasets.MNIST(root='/cnn_data_MNIST', train=True, download=True, transform=transform)

# Importamos el conjunto de datos de prueba de MNIST
test_data = datasets.MNIST(root='/cnn_data_MNIST', train=False, download=True, transform=transform)



# Supongamos que tienes 784 entradas (como imágenes 28x28) y 10 clases
input_size = 784
hidden1 = 500
hidden2 = 300
output_size = 10
learning_rate = 0.01

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)
            # No Softmax aquí: CrossEntropyLoss lo incluye internamente
        )

    def forward(self, x):
        return self.model(x)

# Instancia del modelo
model = SimpleNN()

# Función de pérdida y optimizador (el optimizador es Stochastic Gradient Descent)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)





# Para probar un valor unico en el modelo
# Vemos la imagen número 44 del conjunto
plt.imshow(test_data[44][0].reshape(28, 28)) # Mostramos la imagen
plt.show()   
# Pasamos la imagen por el modelo
with torch.no_grad():
    single_image = test_data[44][0].view(1,1,28,28)  # Batch size 1, 1 canal de color, 28x28 imagen
    output = model(single_image)
print(f'La clase predicha para la imagen 44 es: {output.argmax().item()}')