import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graficas import plot_training_validation_loss, plot_accuracy, plot_confusion_matrix, visualize_image_through_layers
from utils.procesamiento_externo import preprocess_external_image, predict_external_image
from utils.carga_prueba_modelos import save_model, load_model

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
from torch.utils.data import DataLoader, random_split

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

# 10 clases (números del 0 al 9)
output_size = 10
learning_rate = 0.0001
batch_size = 10
epocas = 2

torch.manual_seed(41)  # Fijamos la semilla para reproducibilidad a la hora de usar aleatoriedad.
                       # Lo quitamos despues de realizar las pruebas

# Convertir archivos de imagenes de MNIST a un tensor de 4 dimensiones 
# (número de imágenes, altura, ancho, color)
transform = transforms.ToTensor()   

# Cargamos el conjunto completo de entrenamiento
full_train_data = datasets.MNIST(root='/cnn_data_MNIST', train=True, download=True, transform=transform)

# Definimos tamaños para entrenamiento y validación
train_size = int(0.83334 * len(full_train_data))  # 83,3334% para entrenamiento. Unos 50000
val_size = len(full_train_data) - train_size  # 16,6666% para validación. Unos 10000
# Dividimos el conjunto
train_data, val_data = random_split(full_train_data, [train_size, val_size])

# Importamos el conjunto de datos de prueba de MNIST
test_data = datasets.MNIST(root='/cnn_data_MNIST', train=False, download=True, transform=transform)

# Creamos un tamaño de batch pequeño para las imagenes
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # Suffle true para mezclar los datos
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

rev_dur_epoc = len(train_loader)/10 # Número de batches después del cual imprimir el estado

# Definimos el modelo de CNN
# Este modelo es de un video, ell que se pide es la siguiente clase
class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__() # Inicializa nn.Module
        # Usar padding=1 preserva el tamaño espacial tras la convolución (28x28 -> 28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1) # Capa convolucional 1
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1) # Capa convolucional 2
        self.fc1 = nn.Linear(in_features=7*7*16, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # Aplicamos ReLU después de la primera convolución
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Max pooling. Se reduce a la mitad el tamaño espacial
        # Segunda capa convolucional
        x = F.relu(self.conv2(x)) # Aplicamos ReLU después de la segunda convolución
        x = F.max_pool2d(x, kernel_size=2, stride=2) # Max pooling. Se reduce a la mitad el tamaño espacial
        
        # Aplanamos el tensor para la capa fully connected
        # Con padding=1 en ambas convoluciones y maxpool 2x2 dos veces:
        # 28x28 --conv(pad=1,k=3)--> 28x28 --pool2--> 14x14
        # 14x14 --conv(pad=1,k=3)--> 14x14 --pool2--> 7x7
        # por tanto la representación final tiene tamaño 7*7*16
        x = x.view(-1, 7*7*16) # -1 significa que el tamaño del batch se infiere automáticamente

        # Capas fully connected
        x = F.relu(self.fc1(x)) # Primera capa fully connected con ReLU
        x = F.relu(self.fc2(x)) # Segunda capa fully connected con ReLU
        x = self.fc3(x) # Capa de salida
        #return F.log_softmax(x, dim=1) # Aplicamos log_softmax en la salida. Esta comentado ya que CrossEntropyLoss
                                        # ya incluye log_softmax internamente, por lo que no es necesario aplicarlo aquí.
        return x

# Creamos una instancia del modelo
model = ConvolutionalNN()

# Definimos la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss() # Función de pérdida para clasificación multiclase
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizador Adam

# Para medir el tiempo de entrenamiento, seteamos los tiempos
start_time = time.time()

# Creamos variables para llevar un registro
epochs = epocas
train_losses = []
val_losses = []
train_correct = []
val_correct = []
avg_loss = 0
avg_val_loss = 0
train_accuracies = []
val_accuracies = []

# Hacemos el loop de entrenamiento
for i in range(epochs):
    trn_corr = 0 # Contador de predicciones correctas en entrenamiento
    val_corr = 0 # Contador de predicciones correctas en validacion
    total_loss = 0 
    total_val_loss = 0
    
    # Bucle de entrenamiento
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1  # Contador de batches

        # Aplicamos el modelo
        y_pred = model(X_train) # Obtenemos las predicciones del conjunto de entrenamiento. No está aplanado 2D.

        # Calculamos la pérdida
        loss = criterion(y_pred, y_train)

        # Sumamos el numero de predicciones correctas, indexadas desde el primer punto
        predicted = torch.max(y_pred.data, 1)[1]  # Se usa para obtener la clase predicha por el modelo de clasificación multiclase
        batch_corr = (predicted == y_train).sum() # Cuantos aciertos en el batch
        trn_corr += batch_corr # Acumulamos los aciertos a medida que avanzamos en el entrenamiento

        total_loss += loss.item() # Pérdida en entrenamiento

        # Backpropagation. Actualizamos los pesos
        optimizer.zero_grad()  # Limpiamos los gradientes previos
        loss.backward()        # Calculamos los nuevos gradientes
        optimizer.step()       # Actualizamos los pesos

        # Imprimimos cada rev_dur_epoc batches
        if b % rev_dur_epoc == 0:
            print(f'Epoch: {i}, Batch: {b}, Loss: {loss.item()}, Train Correct: {trn_corr}')

    avg_loss = total_loss / len(train_loader) # Pérdida promedio en entrenamiento
    
    # Porcentajes de aciertos en entrenamiento
    train_acc = trn_corr.item() / len(train_data) * 100

    # Guardamos los porcentajes de aciertos en entrenamiento por época
    train_accuracies.append(train_acc)

    train_losses.append(avg_loss) # Guardamos la pérdida del último batch
    train_correct.append(trn_corr) # Guardamos el número de aciertos en entrenamiento

    # Bucle de validacion
    with torch.no_grad():  # No calculamos gradientes en la fase de validacion
        for X_val, y_val in val_loader:
            y_pred_val = model(X_val) # Obtenemos las predicciones del conjunto de validacion

            predicted = torch.max(y_pred_val.data, 1)[1] # Obtenemos las clases predichas
            val_corr += (predicted == y_val).sum() # Contamos los aciertos en validacion

            loss = criterion(y_pred_val, y_val) # Calculamos la pérdida en validacion
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader) # Pérdida promedio en validacion

    # Porcentaje de aciertos en validacion
    val_acc = val_corr / len(val_data) * 100
    
    # Guardamos los porcentajes de aciertos en validacion por época
    val_accuracies.append(val_acc)

    val_losses.append(avg_val_loss) # Guardamos la pérdida promedio de validacion en la epoca
    val_correct.append(val_corr) # Guardamos el número de aciertos en validacion
    print(f'Epoch: {i}, Validation Correct: {val_corr}, Avg Validation Loss: {avg_val_loss}')

current_time = time.time()
total = current_time - start_time
print(f'Tiempo total de entrenamiento: {total/60} minutos')

# Graficamos las pérdidas de entrenamiento y validacion
plot_training_validation_loss(train_losses, val_losses)

# Graficamos la precisión al final de cada época
plot_accuracy(train_accuracies, val_accuracies)

# Hacemos una prueba con un batch de todos los datos de prueba. Alrededor de 10000 imagenes de prueba
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_loader:
        y_test_pred = model(X_test)
        predicted = torch.max(y_test_pred.data, 1)[1]
        correct += (predicted == y_test).sum()

print(f'\nPrecisión total en el conjunto de prueba: {correct.item()/len(test_data)*100}%')

# Matriz de confusión
# Muesta la cantidad de valores predichos correctamente e incorrectamente por clase
# Por ejemplo, puede mostrar cuántos '3' fueron clasificados como '5', etc.
plot_confusion_matrix(model, test_loader, test_data.targets)

# Para probar un valor unico en el modelo
# Vemos la imagen número 44 del conjunto
'''plt.imshow(test_data[44][0].reshape(28, 28)) # Mostramos la imagen
plt.show()   
# Pasamos la imagen por el modelo
with torch.no_grad():
    single_image = test_data[44][0].view(1,1,28,28)  # Batch size 1, 1 canal de color, 28x28 imagen
    output = model(single_image)
print(f'La clase predicha para la imagen 44 es: {output.argmax().item()}')'''

# Ejemplo: visualizar la imagen con índice 44 del test set (descomentar para probar)
'''example_idx = 44
img_tensor, label = test_data[example_idx]
print(f'\nLabel real de la imagen {example_idx}: {label}')
visualize_image_through_layers(model, img_tensor, idx=example_idx)'''

# Visualizar una imagen externa a través de las capas (descomentar para probar)
'''img_t = preprocess_external_image('../datos_externos_prueba/prueba-2.jpg').to(device='cpu')
visualize_image_through_layers(model, img_t)
# Ejemplo de uso de prediccion de imagen externa (descomentar y ajustar la ruta para probar):
pred, probs = predict_external_image(model, '../datos_externos_prueba/prueba-2.jpg')
print(f'\nPredicción: {pred}, Probabilidades: {probs}')'''

# Para guardar un modelo modelo (no hacer esto cada vez, solo cuando se quiera guardar el mejor modelo)
# save_model(model)

# CARGAR MODELO ENTRENADO Y PROBARLO
# load_model(ConvolutionalNN, "mejor_modelo/mejor_modelo.pth", test_loader, test_data)