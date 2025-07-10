import os
from pathlib import Path

# Configuración de paths 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Configuración IIIT5K Dataset
IIIT5K_DIR = os.path.join(DATA_DIR, 'IIIT5K')
IIIT5K_TRAIN_CSV = os.path.join(IIIT5K_DIR, 'train.csv')
IIIT5K_TEST_CSV = os.path.join(IIIT5K_DIR, 'test.csv')

# Configuración Twitter Dataset
TWITTER_CSV = os.path.join(DATA_DIR, 'twitter', 'training.1600000.processed.noemoticon.csv')

# Configuración de modelos
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

STR_MODEL_PATH = os.path.join(MODELS_DIR, 'crnn_model.h5')
SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_model.h5')