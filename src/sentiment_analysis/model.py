import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

class SentimentAnalyzer:
    def __init__(self, model_path='models/sentiment_model.h5', tokenizer_path='models/tokenizer.pkl'):
        """
        Inicializa el analizador de sentimiento.
        
        Args:
            model_path: Ruta al modelo preentrenado
            tokenizer_path: Ruta al tokenizer guardado
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.max_len = 100  # Debe coincidir con el usado en el entrenamiento
    
    def predict_sentiment(self, text):
        """
        Predice el sentimiento de un texto.
        
        Args:
            text: Texto a analizar
        
        Returns:
            Tupla con (sentimiento, probabilidad)
            Sentimiento puede ser: 'positive', 'negative', 'neutral'
        """
        # Limpiar el texto
        cleaned_text = clean_text(text)
        
        # Convertir a secuencia
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=self.max_len)
        
        # Predecir
        prediction = self.model.predict(padded)[0]
        class_idx = prediction.argmax()
        
        # Mapear índices a sentimientos (depende de cómo entrenaste tu modelo)
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        sentiment = sentiment_map[class_idx]
        confidence = prediction[class_idx]
        
        return sentiment, float(confidence)