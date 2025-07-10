

import os
import cv2
import pandas as pd
import numpy as np
from jiwer import wer, cer
from tqdm import tqdm
from pathlib import Path
from text_recognition.utils import load_str_model, predict_text
from text_recognition.detect_text import preprocess_for_recognition

# ---------------------------------------------------------------
# CONFIGURACIÓN VERIFICADA
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'src' / 'data' / 'IIITK5'
TEST_CSV = DATA_DIR / 'testCharBound.csv'
MODEL_PATH = BASE_DIR / 'src' / 'models' / 'crnn_model.h5'
RESULTS_DIR = BASE_DIR / 'results'

# Ruta exacta donde están las imágenes
IMAGES_DIR = DATA_DIR / 'Words' / 'IIIT5K-Word_V3.0' / 'test'

# ---------------------------------------------------------------
# VERIFICACIÓN INICIAL
# ---------------------------------------------------------------
print("🔍 Verificando archivos...")

if not TEST_CSV.exists():
    available_files = [f.name for f in DATA_DIR.glob('*') if f.is_file()]
    raise FileNotFoundError(f"No se encontró {TEST_CSV}. Archivos disponibles: {available_files}")

if not IMAGES_DIR.exists():
    IMAGES_DIR = DATA_DIR / 'Words' / 'IIIT5K-Word_V3.0' / 'test'
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de imágenes: {IMAGES_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")

RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# FUNCIÓN PARA CARGAR IMÁGENES
# ---------------------------------------------------------------
def load_image(img_name):
    img_path = IMAGES_DIR / img_name
    if not img_path.exists():
        return None, f"Imagen no encontrada: {img_path}"
    
    try:
        img = cv2.imread(str(img_path))
        return img, None if img is not None else "Error al decodificar imagen"
    except Exception as e:
        return None, f"Error al cargar imagen: {str(e)}"

# ---------------------------------------------------------------
# EVALUACIÓN PRINCIPAL
# ---------------------------------------------------------------
print("\n Cargando datos y modelo...")
model = load_str_model(str(MODEL_PATH))
test_df = pd.read_csv(TEST_CSV)

results = []
success_count = 0

print(f"\n Procesando {len(test_df)} imágenes...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_name = row['ImgName'].split('/')[-1]
    true_text = str(row['chars']).strip().upper()
    
    record = {
        'image': img_name,
        'true_text': true_text,
        'predicted': None,
        'cer': np.nan,
        'wer': np.nan,
        'status': 'failed',
        'error': None
    }
    
    # 1. Cargar imagen
    image, error = load_image(img_name)
    if error:
        record['error'] = error
        results.append(record)
        continue
    
    # 2. Preprocesar y predecir
    try:
        processed = preprocess_for_recognition(image)
        predicted = predict_text(model, processed).strip()
        
        record.update({
            'predicted': predicted,
            'cer': cer(true_text, predicted),
            'wer': wer(true_text, predicted),
            'status': 'success'
        })
        success_count += 1
    except Exception as e:
        record['error'] = f"Error en predicción: {type(e).__name__}: {str(e)}"
    
    results.append(record)

# ---------------------------------------------------------------
# REPORTE DE RESULTADOS
# ---------------------------------------------------------------
success_results = [r for r in results if r['status'] == 'success']
failed_results = [r for r in results if r['status'] == 'failed']

print("\n" + "="*60)
print(" RESUMEN DE RESULTADOS".center(60))
print("="*60)

if success_results:
    avg_cer = np.mean([r['cer'] for r in success_results])
    avg_wer = np.mean([r['wer'] for r in success_results])
    
    print(f"\n Éxitos: {len(success_results)} imágenes")
    print(f"• CER promedio: {avg_cer:.4f}")
    print(f"• WER promedio: {avg_wer:.4f}")
    
    print("\n🔍 Ejemplos exitosos:")
    for r in success_results[:3]:
        print(f"Imagen: {r['image']}")
        print(f"Real: '{r['true_text']}'")
        print(f"Pred: '{r['predicted']}'")
        print(f"CER: {r['cer']:.3f} | WER: {r['wer']:.3f}\n")
else:
    print("\nNo se procesaron imágenes correctamente")

if failed_results:
    print(f"\n Fallos: {len(failed_results)} imágenes")
    error_counts = {}
    for r in failed_results:
        error_counts[r['error']] = error_counts.get(r['error'], 0) + 1
    
    print("\nTipos de errores:")
    for error, count in error_counts.items():
        print(f"- {error}: {count} ocurrencias")

# Guardar resultados
results_df = pd.DataFrame(results)
results_csv = RESULTS_DIR / 'evaluation_results.csv'
results_df.to_csv(results_csv, index=False)
print(f"\n Resultados guardados en: {results_csv}")

# Guardar resumen
with open(RESULTS_DIR / 'summary.txt', 'w') as f:
    f.write(f"Imágenes procesadas: {len(results)}\n")
    f.write(f"Éxitos: {len(success_results)}\n")
    f.write(f"Fallos: {len(failed_results)}\n")
    if success_results:
        f.write(f"\nMétricas promedio:\n")
        f.write(f"CER: {np.mean([r['cer'] for r in success_results]):.4f}\n")
        f.write(f"WER: {np.mean([r['wer'] for r in success_results]):.4f}\n")
    if failed_results:
        f.write("\nErrores:\n")
        for error, count in error_counts.items():
            f.write(f"- {error}: {count}\n")

print("\n Proceso completado!")