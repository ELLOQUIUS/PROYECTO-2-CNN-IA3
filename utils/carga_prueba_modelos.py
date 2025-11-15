import os
import torch

def save_model(model, save_dir="mejor_modelo", filename="mejor_modelo.pth"):
    """
    Guarda el estado del modelo en la ruta especificada.

    Parámetros:
    - model: instancia del modelo entrenado (torch.nn.Module)
    - save_dir: carpeta donde se guardará el modelo (por defecto "mejor_modelo")
    - filename: nombre del archivo del modelo (por defecto "mejor_modelo.pth")

    Retorna:
    - mensaje de ruta completa donde se guardó el modelo
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"\nModelo guardado en: {model_path}")

def load_model(model_class, model_path, test_loader, test_data):
    """
    Carga un modelo guardado, lo evalúa sobre el conjunto de prueba y muestra la precisión total.

    Parámetros:
    - model_class: clase del modelo (por ejemplo, ConvolutionalNN)
    - model_path: ruta al archivo .pth con los pesos guardados
    - test_loader: DataLoader con los datos de prueba
    - test_data: dataset original de prueba (para contar el total de muestras)

    Retorna:
    - mensaje de precisión total en porcentaje
    """
    print("\nProbando el modelo guardado...")

    # Cargar el modelo
    loaded_model = model_class()
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()

    # Evaluar precisión
    correct = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_test_pred = loaded_model(X_test)
            predicted = torch.max(y_test_pred.data, 1)[1]
            correct += (predicted == y_test).sum()

    accuracy = correct.item() / len(test_data) * 100
    print(f'\nPrecisión total en el conjunto de prueba cargado: {accuracy:.2f}%')
