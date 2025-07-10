import argparse
import os
from text_recognition.detect_text import detect_text_regions, preprocess_for_recognition
from utils import load_str_model, predict_text
from sentiment_analysis import SentimentAnalyzer
import cv2

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Sistema de Reconocimiento de Texto y Análisis de Sentimiento')
    parser.add_argument('image_path', type=str, help='Ruta a la imagen a procesar')
    args = parser.parse_args()
    
    # 1. Cargar modelos
    print("Cargando modelos...")
    str_model = load_str_model('models/str_model.h5')
    sentiment_analyzer = SentimentAnalyzer()
    
    # 2. Detectar texto en la imagen
    print(f"Procesando imagen: {args.image_path}")
    text_regions, coordinates = detect_text_regions(args.image_path)
    
    if not text_regions:
        print("No se encontraron regiones de texto en la imagen.")
        return
    
    # 3. Reconocer texto en cada región
    recognized_texts = []
    for i, region in enumerate(text_regions):
        # Preprocesar para el modelo STR
        processed = preprocess_for_recognition(region)
        
        # Predecir texto
        text = predict_text(str_model, processed)
        recognized_texts.append(text)
        
        # Opcional: mostrar región y texto reconocido
        print(f"Región {i+1}: {text}")
    
    # Unir todo el texto reconocido
    full_text = ' '.join(recognized_texts)
    print("\nTexto completo reconocido:")
    print(full_text)
    
    # 4. Analizar sentimiento
    sentiment, confidence = sentiment_analyzer.predict_sentiment(full_text)
    print(f"\nSentimiento: {sentiment} (confianza: {confidence:.2f})")
    
    # Opcional: guardar resultados
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar texto reconocido
    with open(os.path.join(output_dir, 'recognized_text.txt'), 'w') as f:
        f.write(full_text)
    
    # Guardar imagen con bounding boxes
    image = cv2.imread(args.image_path)
    for (x, y, w, h) in coordinates:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    output_image_path = os.path.join(output_dir, 'detected_text.jpg')
    cv2.imwrite(output_image_path, image)
    print(f"\nResultados guardados en: {output_dir}")

if __name__ == "__main__":
    main()