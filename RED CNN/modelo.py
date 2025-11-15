# La explicacion de algunas librerias importantes esta en el archivo RED CNN PERSONALIZADA/modelo.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graficas import plot_training_validation_loss, plot_accuracy, plot_confusion_matrix, visualize_image_through_layers
from utils.procesamiento_externo import preprocess_external_image, predict_external_image
from utils.carga_prueba_modelos import save_model, load_model

# 10 clases (números del 0 al 9)
output_size = 10
learning_rate = 0.0001
batch_size = 100
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

rev_dur_epoc = len(train_loader)/10 # Número de batches después del cual imprimir el estado

# Definimos el modelo de CNN
# Este modelo es de un video, ell que se pide es la siguiente clase
class ConvolutionalNNSimple(nn.Module):
    def __init__(self):
        super().__init__() # Inicializa nn.Module
        # Usar padding=1 preserva el tamaño espacial tras la convolución (28x28 -> 28x28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1) # Capa convolucional 1
        self.fc1 = nn.Linear(in_features=28*28*6, out_features=120) # Capa fully connected 1
        self.fc2 = nn.Linear(in_features=120, out_features=output_size) # Capa fully connected 2 (salida)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # Aplicamos ReLU después de la primera convolución CONV + ReLU
        x = x.view(-1, 28*28*6) # -1 significa que el tamaño del batch se infiere automáticamente. Aplanamos para la capa fully connected

        # Capas fully connected
        x = F.relu(self.fc1(x)) # Primera capa fully connected con ReLU. FC + ReLU
        x = self.fc2(x) # Capa de salida. FC
        return x

# Creamos una instancia del modelo
model = ConvolutionalNNSimple()

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
print(f'\nTiempo total de entrenamiento: {total/60} minutos')

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

# Para guardar un modelo modelo (no hacer esto cada vez, solo cuando se quiera guardar el mejor modelo)
# save_model(model)

# CARGAR MODELO ENTRENADO Y PROBARLO
# load_model(ConvolutionalNNSimple, "mejor_modelo/mejor_modelo.pth", test_loader, test_data)