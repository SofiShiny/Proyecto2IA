import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Bidirectional, LSTM, Dense
import tensorflow.keras.backend as K
import numpy as np

def load_str_model(model_path):
    """
    Carga el modelo CRNN exactamente como fue definido originalmente
    """
    # 1. Definir la arquitectura IDÉNTICA al modelo original
    input_tensor = Input(name='input', shape=(32, 128, 1))
    
    # Capas convolucionales
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d')(input_tensor)
    x = MaxPooling2D((2, 2), name='max_pooling2d')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_1')(x)
    x = MaxPooling2D((2, 2), name='max_pooling2d_1')(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
    x = MaxPooling2D((1, 2), name='max_pooling2d_2')(x)  # Pooling especial (1,2)
    
    # Reshape crítico - exactamente como en el modelo original
    x = Reshape((-1, 256), name='reshape')(x)
    
    # Capas recurrentes bidireccionales
    x = Bidirectional(
        LSTM(128, return_sequences=True, name='forward_lstm'), 
        merge_mode='concat', name='bidirectional')(x)
    x = Bidirectional(
        LSTM(128, return_sequences=True, name='forward_lstm_1'), 
        merge_mode='concat', name='bidirectional_1')(x)
    
    # Capa de salida (37 clases según tu configuración)
    output_tensor = Dense(37, activation='linear', name='dense')(x)
    
    # 2. Crear modelo
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # 3. Cargar pesos
    model.load_weights(model_path)
    
    return model

def decode_prediction(prediction, char_map):
    """
    Decodifica la salida del modelo a texto.
    
    Args:
        prediction: Salida del modelo (tensor)
        char_map: Diccionario de mapeo de índices a caracteres
    
    Returns:
        Cadena de texto decodificada
    """
    # Obtener los índices de los caracteres más probables
    input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded = K.ctc_decode(prediction, 
                          input_length=input_length,
                          greedy=True)[0][0]
    indices = K.get_value(decoded)
    
    # Convertir índices a caracteres
    text = ''
    for idx in indices[0]:  # Tomamos el primer elemento del batch
        if idx == -1:  # Carácter de padding
            break
        text += char_map.get(int(idx), '?')  # Asegurarse de que idx es entero
    
    return text

def get_char_map():
    """
    Devuelve el mapeo de índices a caracteres usado en el modelo STR.
    Debes completar esto con el mapeo real usado durante el entrenamiento.
    """
    # Ejemplo básico (debes reemplazarlo con tu mapeo real)
    char_map = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
        25: 'Z',
        26: 'a', 27: 'b', 28: 'c', 29: 'd', 30: 'e',
        31: 'f', 32: 'g', 33: 'h', 34: 'i', 35: 'j',
        36: 'k', 37: 'l', 38: 'm', 39: 'n', 40: 'o',
        41: 'p', 42: 'q', 43: 'r', 44: 's', 45: 't',
        46: 'u', 47: 'v', 48: 'w', 49: 'x', 50: 'y',
        51: 'z',
        52: '0', 53: '1', 54: '2', 55: '3', 56: '4',
        57: '5', 58: '6', 59: '7', 60: '8', 61: '9',
        62: ' ', 63: '-', 64: "'", 65: '.', 66: ',',
        67: '!', 68: '?'
    }
    return char_map

def predict_text(model, processed_image):
    """
    Predice texto a partir de una imagen preprocesada.
    
    Args:
        model: Modelo de reconocimiento de texto
        processed_image: Imagen preprocesada
    
    Returns:
        Texto reconocido
    """
    # Asegurar que la imagen tenga la forma correcta
    if len(processed_image.shape) == 3:
        processed_image = np.expand_dims(processed_image, axis=0)
    
    # Predecir
    prediction = model.predict(processed_image)
    
    # Obtener el mapeo de caracteres
    char_map = get_char_map()
    
    # Decodificar la predicción
    text = decode_prediction(prediction, char_map)
    
    return text