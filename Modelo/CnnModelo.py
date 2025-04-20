import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# — Rutas de guardado —
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR     = os.path.join(BASE_DIR, "Modelos_Entrenados")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_FILE    = os.path.join(SAVE_DIR, "cnn.h5")
WEIGHTS_FILE  = os.path.join(SAVE_DIR, "cnn_pesos.weights.h5")
INDICES_FILE  = os.path.join(SAVE_DIR, "class_indices.json")

def crear_modelo(altura=100, anchura=100, canales=3, clases=4):
    """
    Arquitectura de la CNN:
      • Tres bloques Conv2D + MaxPooling2D para extraer características
      • Flatten para pasar a fully connected
      • Dos capas Dense de 256 neuronas con Dropout 50% para reducir overfitting
      • Capa de salida softmax con 'clases' salidas
    """
    m = Sequential([
        Input(shape=(altura, anchura, canales)),
        Conv2D(16, (3,3), padding="same", activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), padding="same", activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), padding="same", activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(clases, activation="softmax")
    ])
    # Se utiliza 'categorical_crossentropy' por clasificación multiclase,
    # y 'adam' como optimizador balanceado entre velocidad y calidad.
    m.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return m

def entrenar_modelo(
    ruta_train, ruta_val,
    epocas=10,      # Número de epochs: suficiente para converger sin sobreentrenar
    batch_size=16,  # Lote de 16 imágenes por actualización de gradiente
    pasos_val=50,   # Número de batches de validación por epoch
    altura=100, anchura=100,
    extra_callbacks=None
):
    """
    Configura generadores de datos con aumento, entrena la CNN,
    guarda el mejor modelo y sus pesos, y almacena el mapeo clase→índice.
    """
    # Aumento de datos para robustez frente a rotaciones, zooms y flips
    train_datagen = ImageDataGenerator(
        rescale=1/255., rotation_range=20,
        width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, vertical_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1/255.)

    # Flujo desde carpetas: asume subcarpetas por clase
    train_gen = train_datagen.flow_from_directory(
        ruta_train, target_size=(altura, anchura),
        batch_size=batch_size, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        ruta_val, target_size=(altura, anchura),
        batch_size=batch_size, class_mode="categorical"
    )

    # Guardar el mapping clase→índice para decodificar predicciones
    with open(INDICES_FILE, 'w') as f:
        json.dump(train_gen.class_indices, f)

    # Construcción y checkpoint del modelo
    model = crear_modelo(altura, anchura, canales=3, clases=len(train_gen.class_indices))
    chk = ModelCheckpoint(MODEL_FILE, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

    callbacks = [chk]
    if extra_callbacks:
        callbacks += extra_callbacks

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epocas,
        validation_steps=pasos_val,
        callbacks=callbacks,
        verbose=1
    )

    # Guardar pesos finales
    model.save_weights(WEIGHTS_FILE)
    return model, history

def predecir_producto(ruta_imagen, modelo=None, umbral=0.6):
    """
    Predicción con umbral:
      • Si la mayor probabilidad < umbral → devuelve "Otra"
      • Si no, devuelve la clase con mayor probabilidad
      • Umbral elegido en 0.6 para filtrar predicciones poco confiables
    """
    if modelo is None:
        modelo = load_model(MODEL_FILE)
        modelo.load_weights(WEIGHTS_FILE)

    # Recuperar class_indices de disco
    with open(INDICES_FILE) as f:
        class_indices = json.load(f)
    idx_to_class = {v:k for k,v in class_indices.items()}

    # Preprocesar la imagen
    img = load_img(ruta_imagen, target_size=(100,100))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = modelo.predict(arr)[0]
    idx   = int(np.argmax(preds))
    max_p = preds[idx]

    return idx_to_class[idx] if max_p >= umbral else "Otra"

if __name__ == "__main__":
    # Prueba rápida de entrenamiento
    base = os.path.dirname(os.path.abspath(__file__))
    entrenar_modelo(
        os.path.join(base, "..", "Datasets", "entrenamiento"),
        os.path.join(base, "..", "Datasets", "validacion"),
        epocas=10, batch_size=16, pasos_val=50
    )
