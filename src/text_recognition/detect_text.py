import cv2
import numpy as np

def detect_text_regions(image_path, min_width=30, min_height=30):
    """
    Detecta regiones de texto en una imagen usando contornos de OpenCV.
    
    Args:
        image_path: Ruta a la imagen a procesar
        min_width: Ancho mínimo para considerar una región como texto
        min_height: Alto mínimo para considerar una región como texto
    
    Returns:
        Lista de imágenes recortadas con las regiones de texto detectadas
        Lista de coordenadas (x, y, w, h) de las regiones
    """
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por tamaño
    text_regions = []
    coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height:
            # Recortar la región
            region = image[y:y+h, x:x+w]
            text_regions.append(region)
            coordinates.append((x, y, w, h))
    
    return text_regions, coordinates

def preprocess_for_recognition(image, target_size=(128, 32)):
    """
    Preprocesa una imagen para el modelo de reconocimiento de texto.
    
    Args:
        image: Imagen a preprocesar
        target_size: Tamaño objetivo (ancho, alto)
    
    Returns:
        Imagen preprocesada
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Redimensionar manteniendo la relación de aspecto
    h, w = gray.shape
    new_w = int((target_size[1] / h) * w)
    new_w = min(new_w, target_size[0])
    
    resized = cv2.resize(gray, (new_w, target_size[1]), interpolation=cv2.INTER_AREA)
    
    # Rellenar con blanco si es necesario
    if new_w < target_size[0]:
        pad = np.ones((target_size[1], target_size[0] - new_w), dtype=np.uint8) * 255
        processed = np.hstack([resized, pad])
    else:
        processed = resized
    
    # Normalizar
    processed = processed.astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=-1)
    
    return processed