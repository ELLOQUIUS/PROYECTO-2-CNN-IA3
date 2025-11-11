# La explicacion de algunas librerias importantes esta en el archivo RED CNN PERSONALIZADA/modelo.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import seaborn as sns
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 10 clases (números del 0 al 9)
output_size = 10
learning_rate = 0.001
batch_size = 10
epocas = 30

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Optimizador SGD (Stochastic Gradient Descent)

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

    # Guardmos los porcentajes de aciertos en entrenamiento por época
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
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.show() 

# Graficamos la precisión al final de cada época
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy al final de cada epoca')
plt.xlabel('Epoch')
plt.ylabel('Percent of Correct Predictions')
plt.legend()
plt.grid(True)
plt.show()

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
all_preds = torch.tensor([])
with torch.no_grad():
    for X_test, _ in test_loader:
        y_pred_test = model(X_test)
        predicted = torch.max(y_pred_test.data, 1)[1]
        all_preds = torch.cat((all_preds, predicted), dim=0)
cm = confusion_matrix(test_data.targets, all_preds.numpy())
df_cm = pd.DataFrame(cm, index=range(10), columns=range(10  ))
plt.figure(figsize=(10,7))
plt.title('Matriz de Confusión')
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Visualización: mostrar cómo una imagen del conjunto de prueba se transforma
def plot_feature_maps(maps, title=None, max_cols=6):
    """Dibuja las feature maps (maps puede ser tensor CxHxW o 1xCxHxW).
    Cada canal se muestra como una mini-imagen en una cuadrícula."""
    maps = maps.detach().cpu()
    if maps.dim() == 4 and maps.size(0) == 1:
        maps = maps.squeeze(0)
    # ahora maps tiene shape (C, H, W)
    C = maps.shape[0]
    cols = min(C, max_cols)
    rows = (C + cols - 1) // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(C):
        plt.subplot(rows, cols, i + 1)
        fm = maps[i]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6)
        plt.imshow(fm, cmap='viridis')
        plt.axis('off')
    if title:
        plt.suptitle(title)
    plt.show()

def visualize_image_through_layers(model, image_tensor, idx=None):
    """Muestra la imagen original y las activaciones después de conv1, pool1, conv2, pool2.
    image_tensor debe ser shape (1,28,28) o (28,28)."""
    model.eval()
    with torch.no_grad():
        img = image_tensor.clone()
        if img.dim() == 2:
            img = img.unsqueeze(0)
        img_batch = img.view(1, 1, 28, 28)

        a1 = F.relu(model.conv1(img_batch))
        #p1 = F.max_pool2d(a1, kernel_size=2, stride=2)
        #a2 = F.relu(model.conv2(p1))
        #p2 = F.max_pool2d(a2, kernel_size=2, stride=2)

        # Mostrar original
        plt.figure(figsize=(3,3))
        plt.imshow(img.squeeze(0).cpu().numpy(), cmap='gray')
        title = f'Original'
        if idx is not None:
            title += f' (idx={idx})'
        plt.title(title)
        plt.axis('off')
        plt.show()

        # Mostrar activaciones
        plot_feature_maps(a1.squeeze(0), title='After conv1 (ReLU)')
        #plot_feature_maps(p1.squeeze(0), title='After pool1')
        #plot_feature_maps(a2.squeeze(0), title='After conv2 (ReLU)')
        #plot_feature_maps(p2.squeeze(0), title='After pool2')

# Ejemplo: visualizar la imagen con índice 44 del test set
example_idx = 44
img_tensor, label = test_data[example_idx]
print(f'\nLabel real de la imagen {example_idx}: {label}')
visualize_image_through_layers(model, img_tensor, idx=example_idx)