import h5py
with h5py.File('src/models/crnn_model.h5', 'r') as f:
    print("Configuración del modelo:", f.attrs.get('model_config', 'No disponible'))