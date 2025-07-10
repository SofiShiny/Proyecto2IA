import argparse
import os
from src.text_recognition.detect_text import detect_text_regions, preprocess_for_recognition
from src.text_recognition.utils import load_str_model, predict_text
from src.sentiment_analysis import SentimentAnalyzer
import cv2
import warnings

# Suprimir warnings de TensorFlow y NLTK
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

def main():
    # Configurar argumentos de línea de comandos mejorado
    parser = argparse.ArgumentParser(description='Sistema de Reconocimiento de Texto y Análisis de Sentimiento')
    parser.add_argument('--input_image', type=str, required=True, help='Ruta a la imagen a procesar')
    parser.add_argument('--output_dir', type=str, default='output', help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Descargar recursos de NLTK solo si no existen
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Descargando recursos de NLTK...")
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # 1. Cargar modelos
    print("Cargando modelos...")
    str_model = load_str_model('src/models/crnn_model.h5')
    sentiment_analyzer = SentimentAnalyzer()
    
    # 2. Procesar imagen
    print(f"\nProcesando imagen: {args.input_image}")
    text_regions, coordinates = detect_text_regions(args.input_image)
    
    if not text_regions:
        print("No se encontraron regiones de texto en la imagen.")
        return
    
    # 3. Reconocer texto
    recognized_texts = []
    for i, (region, (x, y, w, h)) in enumerate(zip(text_regions, coordinates)):
        processed = preprocess_for_recognition(region)
        text = predict_text(str_model, processed)
        recognized_texts.append(text)
        print(f"Región {i+1} ({w}x{h}px): {text}")
    
    full_text = ' '.join(recognized_texts)
    
    # 4. Analizar sentimiento
    sentiment, confidence = sentiment_analyzer.predict_sentiment(full_text)
    
    # 5. Guardar resultados
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Texto reconocido
    text_path = os.path.join(args.output_dir, 'recognized_text.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"TEXTO RECONOCIDO:\n{full_text}\n\nSENTIMIENTO: {sentiment} (Confianza: {confidence:.2%})")
    
    # Imagen con bounding boxes
    image = cv2.imread(args.input_image)
    for (x, y, w, h) in coordinates:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    image_path = os.path.join(args.output_dir, 'output_image.jpg')
    cv2.imwrite(image_path, image)
    
    print(f"""
    \nRESULTADOS:
    • Texto reconocido: {full_text}
    • Sentimiento: {sentiment} ({confidence:.2%})
    • Archivos guardados en: {args.output_dir}
        - Texto: {text_path}
        - Imagen: {image_path}
    """)

if __name__ == "__main__":
    main()