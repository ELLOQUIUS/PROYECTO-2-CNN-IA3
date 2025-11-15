import numpy as np
from PIL import Image
import torch

# ---------- Helpers para usar imágenes externas ----------
def preprocess_external_image(path, invert_if_needed=True):
    """Carga una imagen desde `path` y la convierte a tensor 1x1x28x28 listo para el modelo.

    Pasos:
    - Abre y convierte a escala de grises
    - Redimensiona a 28x28 (antialias)
    - Normaliza a [0,1]
    - Invierte los valores si la imagen parece tener fondo claro (para coincidir con MNIST)
    - Devuelve un tensor torch.float32 shape (1,1,28,28)
    """
    img = Image.open(path).convert('L')  # L = grayscale
    # Redimensionar a 28x28 (ANTIALIAS para mejor calidad)
    try:
        img = img.resize((28, 28), Image.ANTIALIAS)
    except Exception:
        img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0  # escala 0..1

    # En MNIST el fondo es oscuro (cercano a 0) y la cifra clara (cercana a 1).
    # Si la foto tiene fondo claro (media alta), invertimos.
    if invert_if_needed:
        if arr.mean() > 0.5:
            arr = 1.0 - arr

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1 x 1 x 28 x 28
    return tensor.type(torch.float32)

def predict_external_image(model, image_path, device='cpu'):
    """Preprocesa `image_path`, pasa por `model` y devuelve (predicción, probs).

    - pred: entero con la clase (0..9)
    - probs: array numpy con probabilidades por clase
    """
    model.eval()
    img_t = preprocess_external_image(image_path).to(device)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        #print(f'\nLa clase predicha para la imagen es: {logits.argmax().item()}')
    return pred, probs.squeeze(0).cpu().numpy()