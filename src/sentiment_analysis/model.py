import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_path='src/models/crnn_model.h5'):
        """
        Inicializa el analizador con modelo y crea un tokenizer nuevo.
        
        Args:
            model_path: Ruta al modelo (.h5).
        """
        try:
            # Carga el modelo
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer='adam', 
                             loss='categorical_crossentropy', 
                             metrics=['accuracy'])
            
            # Crea un tokenizer nuevo (vacío)
            self.tokenizer = Tokenizer(num_words=10000)  # Ajusta según necesidades
            self.tokenizer.word_index = {"<PAD>": 0, "<UNK>": 1}  # Vocabulario mínimo
            
        except Exception as e:
            print(f"Error al inicializar: {e}")
            raise
        
        self.max_len = 100  # Debe coincidir con el usado en el entrenamiento

    def update_tokenizer(self, texts):
        """
        Actualiza el tokenizer con nuevos textos.
        Útil si no tenías un tokenizer pre-entrenado.
        
        Args:
            texts: Lista de textos para entrenar el tokenizer (ej: ["texto 1", "texto 2"])
        """
        self.tokenizer.fit_on_texts(texts)

    def preprocess_text(self, text):
        """
        Convierte texto crudo a secuencia numérica.
        """
        sequences = self.tokenizer.texts_to_sequences([text])
        return pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

    def predict_sentiment(self, text):
        """
        Predice sentimiento a partir de texto crudo.
        """
        try:
            # Preprocesamiento automático
            preprocessed = self.preprocess_text(text)
            
            # Predicción
            prediction = self.model.predict(preprocessed)[0]
            class_idx = np.argmax(prediction)
            
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Ajusta esto
            return sentiment_map[class_idx], float(prediction[class_idx])
            
        except Exception as e:
            print(f"Error al predecir: {e}")
            return "error", 0.0