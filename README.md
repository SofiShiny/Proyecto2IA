# Proyecto de IA: Sistema de Reconocimiento de Texto y Análisis de Sentimiento

## Visión General

Este proyecto implementa un sistema de inteligencia artificial de dos etapas que:

1. **Reconoce texto en imágenes**: Utiliza un modelo CNN-RNN entrenado con el dataset IIIT-5K para extraer texto de imágenes como carteles, anuncios o capturas de pantalla.
2. **Analiza el sentimiento**: Procesa el texto extraído con un modelo RNN entrenado para clasificar el contenido como positivo, negativo o neutral.

## Arquitectura del Sistema

### Pipeline Completo:
```
Imagen → Detección de texto → Reconocimiento de texto → Análisis de sentimiento → Resultado
```

### Componentes Clave:

1. **Módulo STR (Scene Text Recognition)**:
   - Modelo CNN-RNN para reconocimiento óptico de caracteres
   - Preprocesamiento de imágenes con OpenCV
   - Métricas de evaluación: CER (Character Error Rate) y WER (Word Error Rate)

2. **Módulo de Análisis de Sentimiento**:
   - Modelo RNN con capas LSTM/GRU
   - Pipeline de preprocesamiento de texto
   - Clasificación en 3 categorías: positivo, neutral, negativo

## Guía de Implementación

### Fase 0: Configuración Inicial

```bash
# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  

# Instalar dependencias
pip install tensorflow opencv-python pandas scikit-learn jiwer
```

## 💡 Mejoras Futuras

1. Implementar detección de texto multilingüe
2. Añadir soporte para análisis de sentimiento por frases
3. Desarrollar interfaz gráfica de usuario

**Nota**: Para ejecutar el proyecto completo debe descargar los datasets requeridos

Comandos: 
1. python src/test_str_model.py   
2. python main.py --input_image ejemplos/image9425.jpeg --output_dir resultados/                                                                