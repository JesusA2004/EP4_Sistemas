# ——————————————————————————————
#             Importaciones
# ——————————————————————————————
import os                                 # Manejo de rutas y archivos en el sistema
import json                               # Serialización de índices de clase
import numpy as np                        # Cálculos numéricos y manipulación de arrays
from PIL import Image, UnidentifiedImageError  # Apertura y validación de imágenes
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.callbacks import ModelCheckpoint  # Para guardar el mejor modelo durante el fit


# ——————————————————————————————
#          Rutas de guardado de modelo
# ——————————————————————————————
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR     = os.path.join(BASE_DIR, "Modelos_Entrenados")
# Asegura que exista la carpeta donde se guardarán pesos, arquitectura e índices
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_FILE    = os.path.join(SAVE_DIR, "cnn.h5")                # Archivo con arquitectura + pesos
WEIGHTS_FILE  = os.path.join(SAVE_DIR, "cnn_pesos.weights.h5")  # Solo pesos finales
INDICES_FILE  = os.path.join(SAVE_DIR, "class_indices.json")    # Mapeo clase→índice para decodificar predicciones


# ——————————————————————————————
#   Limpieza y conversión de imágenes corruptas o no-JPEG
# ——————————————————————————————
def _clean_and_convert_images(root_dir):
    """
    Recorre de forma recursiva el directorio raíz:
      • Verifica que cada archivo sea una imagen válida (se puede abrir con PIL).
      • Si es corrupta o no se puede abrir, la elimina.
      • Si no tiene extensión .jpg/.jpeg, la convierte a JPEG real y borra el original.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            base, ext = os.path.splitext(full_path)
            ext_lower = ext.lower()
            
            # Intento de apertura y verificación de integridad
            try:
                with Image.open(full_path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                print(f"[WARNING] Imagen inválida, eliminando: {full_path}")
                os.remove(full_path)
                continue

            # Conversión a JPEG si no tiene extensión adecuada
            if ext_lower not in ('.jpg', '.jpeg'):
                try:
                    with Image.open(full_path).convert('RGB') as im:
                        new_path = base + '.jpg'
                        im.save(new_path, 'JPEG')
                    os.remove(full_path)
                    print(f"[INFO] Convertida: {full_path} → {new_path}")
                except Exception as e:
                    print(f"[WARNING] No se pudo convertir {full_path}: {e}")
                    os.remove(full_path)


# ——————————————————————————————
#              Definición de la CNN
# ——————————————————————————————
def crear_modelo(altura=100, anchura=100, canales=3, clases=4):
    """
    Construye y compila una red neuronal convolucional (CNN) con:
      • 3 bloques de Conv2D + MaxPooling2D para extraer características.
      • Flatten para pasar a fully connected.
      • 2 capas Dense de 256 unidades con Dropout 50% para reducir overfitting.
      • Capa de salida softmax con 'clases' neuronas (clasificación multiclase).
    Parámetros:
      - altura, anchura: dimensiones de entrada de la imagen.
      - canales: número de canales (3 para RGB).
      - clases: cantidad de categorías a predecir.
    Retorna:
      - modelo compilado con 'categorical_crossentropy' y optimizador 'adam'.
    """
    model = Sequential([
        Input(shape=(altura, anchura, canales)),
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(clases, activation="softmax")
    ])
    # 'categorical_crossentropy' es apropiada para más de dos clases,
    # y 'adam' equilibra velocidad de convergencia y estabilidad.
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


# ——————————————————————————————
#         Entrenamiento de la CNN
# ——————————————————————————————
def entrenar_modelo(
    ruta_train, ruta_val,
    epocas=10,      # Número de pasadas completas por el dataset
    batch_size=16,  # Tamaño de lote para cada paso de gradiente
    pasos_val=50,   # Pasos de validación por época
    altura=100, anchura=100,
    extra_callbacks=None
):
    """
    Prepara generadores de imagen con data augmentation, entrena la CNN,
    guarda el mejor modelo según val_accuracy y almacena el mapping clase→índice.
    Parámetros:
      - ruta_train, ruta_val: carpetas con subcarpetas por clase.
      - epocas, batch_size, pasos_val: controlan el proceso de fit.
      - altura, anchura: redimensionamiento de imagen.
      - extra_callbacks: lista de callbacks adicionales (p.ej. barra de progreso).
    Retorna:
      - modelo entrenado y objeto history con métricas de training/validation.
    """
    # 1) Limpieza y conversión de todo el dataset
    _clean_and_convert_images(ruta_train)
    _clean_and_convert_images(ruta_val)

    # 2) Generadores de datos con aumento (solo para entrenamiento)
    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1/255.)

    # 3) Flujo de carpetas: detecta subcarpetas como clases
    train_gen = train_datagen.flow_from_directory(
        ruta_train,
        target_size=(altura, anchura),
        batch_size=batch_size,
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        ruta_val,
        target_size=(altura, anchura),
        batch_size=batch_size,
        class_mode="categorical"
    )

    # 4) Guardar índices de clases para decodificar predicciones
    with open(INDICES_FILE, 'w') as f:
        json.dump(train_gen.class_indices, f)

    # 5) Crear el modelo y configurar checkpoint
    model = crear_modelo(
        altura=altura,
        anchura=anchura,
        canales=3,
        clases=len(train_gen.class_indices)
    )
    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Combinar callbacks propios y adicionales
    callbacks = [checkpoint]
    if extra_callbacks:
        callbacks += extra_callbacks

    # 6) Entrenar con validación en cada época
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epocas,
        validation_steps=pasos_val,
        callbacks=callbacks,
        verbose=1
    )

    # 7) Guardar pesos finales (por si el checkpoint no cubrió todo)
    model.save_weights(WEIGHTS_FILE)

    return model, history


# ——————————————————————————————
#      Predicción con umbral de confianza
# ——————————————————————————————
def predecir_producto(ruta_imagen, modelo=None, umbral=0.6):
    """
    Carga un modelo entrenado (si no se proporciona), prepara la imagen,
    obtiene probabilidades y aplica un umbral mínimo:
      • Si la probabilidad máxima < umbral → devuelve "Otra".
      • Si no, devuelve la clase con mayor probabilidad.
    Parámetros:
      - ruta_imagen: ruta al archivo a predecir.
      - modelo: instancia cargada o None para cargar desde disco.
      - umbral: confianza mínima aceptable para devolver una clase.
    """
    # 1) Cargar modelo y pesos de disco si es necesario
    if modelo is None:
        modelo = load_model(MODEL_FILE)
        modelo.load_weights(WEIGHTS_FILE)

    # 2) Recuperar mapping índice→clase
    with open(INDICES_FILE) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    # 3) Preprocesamiento de la imagen
    img = load_img(ruta_imagen, target_size=(100, 100))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)  # Añadir dimensión de batch

    # 4) Predicción y decisión con umbral
    preds = modelo.predict(arr)[0]
    idx = int(np.argmax(preds))
    max_prob = preds[idx]

    return idx_to_class[idx] if max_prob >= umbral else "Otra"


# ——————————————————————————————
#        Punto de entrada para pruebas
# ——————————————————————————————
if __name__ == "__main__":
    # Ejecución de prueba: entrena con los datasets especificados
    base = os.path.dirname(os.path.abspath(__file__))
    entrenar_modelo(
        ruta_train=os.path.join(base, "..", "Datasets", "entrenamiento"),
        ruta_val=os.path.join(base, "..", "Datasets", "validacion"),
        epocas=10,
        batch_size=16,
        pasos_val=50
    )
