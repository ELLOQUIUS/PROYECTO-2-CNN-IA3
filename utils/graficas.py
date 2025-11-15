import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import numpy as np

def plot_training_validation_loss(train_losses, val_losses):
    """
    Grafica las pérdidas de entrenamiento y validación a lo largo de las épocas.

    Parámetros:
    - train_losses: lista de pérdidas de entrenamiento por época (floats)
    - val_losses: lista de pérdidas de validación por época (floats)

    Retorna:
    - None (muestra la gráfica)
    """
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies):
    """
    Grafica la precisión de entrenamiento y validación al final de cada época.

    Parámetros:
    - train_accuracies: lista de precisiones de entrenamiento por época (en porcentaje)
    - val_accuracies: lista de precisiones de validación por época (en porcentaje)

    Retorna:
    - None (muestra la gráfica)
    """
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy al final de cada época')
    plt.xlabel('Epoch')
    plt.ylabel('Percent of Correct Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(model, test_loader, test_targets, num_classes=10):
    """
    Calcula y grafica la matriz de confusión para un modelo dado.

    Parámetros:
    - model: red neuronal entrenada
    - test_loader: DataLoader con los datos de prueba
    - test_targets: etiquetas reales de los datos de prueba (tensor)
    - num_classes: número de clases (por defecto 10)

    Retorna:
    - None (muestra la gráfica)
    """
    model.eval()
    all_preds = torch.tensor([])

    with torch.no_grad():
        for X_test, _ in test_loader:
            y_pred_test = model(X_test)
            predicted = torch.max(y_pred_test.data, 1)[1]
            all_preds = torch.cat((all_preds, predicted), dim=0)

    cm = confusion_matrix(test_targets, all_preds.numpy())
    df_cm = pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes))

    plt.figure(figsize=(10, 7))
    plt.title('Matriz de Confusión')
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Clases reales')
    plt.ylabel('Clases predichas')
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
        p1 = F.max_pool2d(a1, kernel_size=2, stride=2)
        a2 = F.relu(model.conv2(p1))
        p2 = F.max_pool2d(a2, kernel_size=2, stride=2)

        # Mostrar original
        plt.figure(figsize=(3,3))

        # Esto es para imágenes externas al conjunto MNIST
        # img puede tener shape (28,28), (1,28,28) o (1,1,28,28).
        disp = img.squeeze().cpu().numpy()
        # Si queda como (C,H,W) con C==3, transponer a (H,W,3) para imshow;
        # si C==1, quitar la dimensión de canal.
        if disp.ndim == 3:
            if disp.shape[0] == 3:
                disp = np.transpose(disp, (1, 2, 0))
            elif disp.shape[0] == 1:
                disp = disp.squeeze(0)


        plt.imshow(disp, cmap='gray')
        title = f'Original'
        if idx is not None:
            title += f' (idx={idx})'
        plt.title(title)
        plt.axis('off')
        plt.show()

        # Mostrar activaciones
        plot_feature_maps(a1.squeeze(0), title='After conv1 (ReLU)')
        plot_feature_maps(p1.squeeze(0), title='After pool1')
        plot_feature_maps(a2.squeeze(0), title='After conv2 (ReLU)')
        plot_feature_maps(p2.squeeze(0), title='After pool2')