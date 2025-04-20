import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# — Rutas de guardado —
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR      = os.path.join(BASE_DIR, "Modelos_Entrenados")
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_FILE    = os.path.join(SAVE_DIR, "cnn.h5")
WEIGHTS_FILE  = os.path.join(SAVE_DIR, "cnn_pesos.weights.h5")
INDICES_FILE  = os.path.join(SAVE_DIR, "class_indices.json")

def crear_modelo(altura=50, anchura=50, canales=3, clases=3,
                 kernels1=16, kernels2=32,
                 kernel1_size=(3,3), kernel2_size=(3,3),
                 size_pooling=(3,3)):
    model = Sequential([
        Input(shape=(altura, anchura, canales)),
        Conv2D(kernels1, kernel1_size, padding="same", activation="relu"),
        MaxPooling2D(pool_size=size_pooling),
        Conv2D(kernels2, kernel2_size, padding="same", activation="relu"),
        MaxPooling2D(pool_size=size_pooling),
        Flatten(),
        Dense(255, activation="relu"),
        Dense(255, activation="relu"),
        Dropout(0.5),
        Dense(clases, activation="softmax")
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "mse"]
    )
    return model

def entrenar_modelo(ruta_entrenamiento, ruta_validacion,
                    epocas=100, batch_size=2, pasos=100,
                    altura=50, anchura=50):
    # — Generadores de imágenes —
    train_datagen = ImageDataGenerator(
        rescale=1/255, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True,
        vertical_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1/255)

    train_gen = train_datagen.flow_from_directory(
        ruta_entrenamiento,
        target_size=(altura, anchura),
        batch_size=batch_size,
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        ruta_validacion,
        target_size=(altura, anchura),
        batch_size=batch_size,
        class_mode="categorical"
    )

    # — Guardar mapping clase→índice en un JSON —
    class_indices = train_gen.class_indices  # e.g. {'Boing':0,'CocaCola':1,'Jarrito':2}
    with open(INDICES_FILE, 'w') as f:
        json.dump(class_indices, f)

    # — Construir y compilar —
    model = crear_modelo(altura, anchura, clases=len(class_indices))

    # — Callback para guardar el mejor modelo —
    chk = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # — Entrenar —
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epocas,
        validation_steps=pasos,
        callbacks=[chk],
        verbose=1
    )

    # — Guardar pesos aparte —
    model.save_weights(WEIGHTS_FILE)

    return model, history

def predecir_producto(ruta_imagen, modelo=None, umbral=0.6):
    # — Cargar modelo y pesos si no se pasó —
    if modelo is None:
        modelo = load_model(MODEL_FILE)
        modelo.load_weights(WEIGHTS_FILE)

    # — Cargar mapping índice→clase —
    with open(INDICES_FILE, 'r') as f:
        class_indices = json.load(f)          # e.g. {'Boing':0,'CocaCola':1,'Jarrito':2}
    idx_to_class = {v:k for k,v in class_indices.items()}

    # — Preprocesar imagen —
    img = load_img(ruta_imagen, target_size=(50,50))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # — Predecir —
    preds = modelo.predict(arr)[0]           # array([p0, p1, p2])
    idx   = int(np.argmax(preds))
    max_prob = float(preds[idx])

    # — Decisión con umbral —
    if max_prob < umbral:
        return "Otra"
    return idx_to_class[idx]

if __name__ == "__main__":
    # Prueba rápida de entrenamiento
    base = os.path.dirname(os.path.abspath(__file__))
    ent  = os.path.join(base, "..", "Datasets", "entrenamiento")
    val  = os.path.join(base, "..", "Datasets", "validacion")
    entrenar_modelo(ent, val, epocas=5)
