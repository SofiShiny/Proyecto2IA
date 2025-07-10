import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, 'data', 'IIITK5'))
IIIT5K_DIR = os.path.normpath(os.path.join(DATA_DIR, 'Words', 'IIIT5K-Word_V3.0'))
TRAIN_CSV = os.path.normpath(os.path.join(DATA_DIR, 'traindata.csv'))
TEST_CSV = os.path.normpath(os.path.join(DATA_DIR, 'testdata.csv'))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, 'models', 'crnn_model.h5'))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def debug_data_shapes(X, y, name="Dataset"):
    print(f"\nDebug {name}:")
    print(f"- Número de muestras: {len(X)}")
    print(f"- Shape imágenes: {X[0].shape if len(X) > 0 else 'N/A'}")
    print(f"- Ejemplo label: {y[0] if len(y) > 0 else 'N/A'}")
    print(f"- Tipos de datos - X: {type(X[0]) if len(X) > 0 else 'N/A'}, y: {type(y[0]) if len(y) > 0 else 'N/A'}")

def load_data(csv_path, images_dir):
    print(f"\nCargando datos desde: {csv_path}")
    data = pd.read_csv(csv_path)
    images = []
    labels = []
    
    for idx, row in data.iterrows():
        img_name = os.path.normpath(row['ImgName']).replace('\\', os.sep).replace('/', os.sep)
        img_name = os.path.basename(img_name)
        img_path = os.path.normpath(os.path.join(images_dir, img_name))
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: No se pudo cargar {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(row['GroundTruth'])
        except Exception as e:
            print(f"Error procesando {img_path}: {str(e)}")
    
    debug_data_shapes(images, labels, "Datos crudos")
    return images, labels

def preprocess_image(img, img_size=(128, 32)):
    try:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        h, w = img.shape
        new_w = int(img_size[1] * w / h)
        img = cv2.resize(img, (new_w, img_size[1]), interpolation=cv2.INTER_CUBIC)
        
        if new_w < img_size[0]:
            img = cv2.copyMakeBorder(img, 0, 0, 0, img_size[0]-new_w, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(img, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
        
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=-1)
    except Exception as e:
        print(f"Error en preprocesamiento: {str(e)}")
        return None

def ctc_loss(y_true, y_pred):
    """Función de pérdida CTC con verificación de tipos"""
    # Verificación de tipos y shapes
    print("\nDebug CTC Loss:")  # Solo para depuración
    print(f"y_true dtype: {y_true.dtype}, shape: {y_true.shape}")
    print(f"y_pred dtype: {y_pred.dtype}, shape: {y_pred.shape}")
    
    # Convertir tipos si es necesario
    y_true = tf.cast(y_true, tf.int32)
    
    # Obtener longitudes de secuencia
    input_length = tf.math.reduce_sum(tf.ones_like(y_pred[:, :, 0]), axis=1)
    label_length = tf.math.reduce_sum(tf.ones_like(y_true), axis=1)
    
    # Verificación adicional
    print(f"Input length: {input_length[:5]}")
    print(f"Label length: {label_length[:5]}")
    print(f"Unique labels: {tf.unique(tf.reshape(y_true, [-1]))[0]}")
    
    # Calcular pérdida CTC
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=0
    )
    
    return tf.reduce_mean(loss)

def build_crnn_model(input_shape, num_chars):
    input_img = Input(shape=input_shape, name='input')
    
    # Capas CNN
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2))(x)
    
    # Redimensionar para LSTM
    x = Reshape((-1, 256))(x)
    
    # Capas LSTM Bidireccionales
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Capa de salida (num_chars + 1 para blank)
    output = Dense(num_chars + 1)(x)
    
    model = Model(inputs=input_img, outputs=output)
    
    print("\nResumen del modelo:")
    model.summary()
    
    return model

def encode_labels(labels, char_to_num, max_length=20):
    """Codificación con verificación exhaustiva"""
    encoded = []
    unknown_chars = set()
    
    for label in labels:
        label_str = str(label).upper().strip()
        seq = []
        
        for c in label_str:
            if c in char_to_num:
                seq.append(char_to_num[c])
            else:
                unknown_chars.add(c)
                seq.append(0)  # Usar blank para caracteres desconocidos
        
        # Verificar longitud
        if len(seq) > max_length:
            seq = seq[:max_length]
        else:
            seq = seq + [0] * (max_length - len(seq))
        
        encoded.append(seq)
    
    if unknown_chars:
        print(f"\nAdvertencia: Caracteres desconocidos encontrados: {unknown_chars}")
    
    encoded_array = np.array(encoded, dtype=np.int32)
    
    print("\nDebug encode_labels:")
    print(f"- Ejemplo label original: {labels[0]}")
    print(f"- Ejemplo label codificado: {encoded_array[0]}")
    print(f"- Shape final: {encoded_array.shape}")
    print(f"- Valores únicos: {np.unique(encoded_array)}")
    print(f"- Tipo de datos: {encoded_array.dtype}")
    
    return encoded_array

def main():
    # Desactivar oneDNN si causa problemas
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    try:
        # 1. Cargar datos
        print("\n" + "="*50)
        print("Paso 1: Cargando datos...")
        train_images, train_labels = load_data(TRAIN_CSV, os.path.join(IIIT5K_DIR, 'train'))
        test_images, test_labels = load_data(TEST_CSV, os.path.join(IIIT5K_DIR, 'test'))
        
        # 2. Preprocesar imágenes
        print("\n" + "="*50)
        print("Paso 2: Preprocesando imágenes...")
        X_train = []
        for img in train_images:
            processed = preprocess_image(img)
            if processed is not None:
                X_train.append(processed)
        X_train = np.array(X_train)
        
        X_test = []
        for img in test_images:
            processed = preprocess_image(img)
            if processed is not None:
                X_test.append(processed)
        X_test = np.array(X_test)
        
        print("\nDebug preprocesamiento:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Rango de valores: {np.min(X_train)} - {np.max(X_train)}")
        
        # 3. Crear mapeo de caracteres
        print("\n" + "="*50)
        print("Paso 3: Creando mapeo de caracteres...")
        all_chars = set()
        for label in train_labels + test_labels:
            all_chars.update(str(label).upper().strip())
        
        sorted_chars = sorted(all_chars)
        char_to_num = {c: i+1 for i, c in enumerate(sorted_chars)}
        num_to_char = {i+1: c for i, c in enumerate(sorted_chars)}
        num_chars = len(char_to_num)
        
        print(f"\nVocabulario ({num_chars} caracteres): {''.join(sorted_chars)}")
        print(f"Ejemplo de mapeo: 'A' -> {char_to_num.get('A', 'No encontrado')}")
        
        # 4. Codificar etiquetas
        print("\n" + "="*50)
        print("Paso 4: Codificando etiquetas...")
        max_label_length = max(len(str(label)) for label in train_labels + test_labels)
        print(f"Longitud máxima de etiqueta: {max_label_length}")
        
        y_train = encode_labels(train_labels, char_to_num, max_length=max_label_length)
        y_test = encode_labels(test_labels, char_to_num, max_length=max_label_length)
        
        # Verificación final antes del entrenamiento
        print("\n" + "="*50)
        print("Verificación final antes del entrenamiento:")
        print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
        print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
        print(f"Máximo valor en y_train: {np.max(y_train)}, debería ser <= {num_chars}")
        
        # 5. Construir y compilar modelo
        print("\n" + "="*50)
        print("Paso 5: Construyendo modelo...")
        model = build_crnn_model((32, 128, 1), num_chars)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=ctc_loss)
        
        # 6. Callbacks
        callbacks = [
            ModelCheckpoint(
                MODEL_PATH,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # 7. Entrenamiento
        print("\n" + "="*50)
        print("Paso 6: Iniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\nModelo guardado en {MODEL_PATH}")
    
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        print("\nInformación de depuración:")
        if 'X_train' in locals():
            print(f"- X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        if 'y_train' in locals():
            print(f"- y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            print(f"- Valores únicos en y_train: {np.unique(y_train)}")
        if 'char_to_num' in locals():
            print(f"- Tamaño del vocabulario: {len(char_to_num)}")
        
        print("\nSolución sugerida:")
        print("1. Verifica que todas las imágenes se cargaron correctamente")
        print("2. Asegúrate que todos los caracteres en las etiquetas están en el vocabulario")
        print("3. Verifica que las etiquetas codificadas sean de tipo entero (int32)")
        print("4. Prueba con un batch_size más pequeño si persisten los problemas")

if __name__ == "__main__":
    main()