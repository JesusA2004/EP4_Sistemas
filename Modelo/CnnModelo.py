# model/cnn_model.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

def crear_modelo(altura=50, anchura=50, canales=3, clases=3,
                 kernels1=16, kernels2=32, kernel1_size=(3,3),
                 kernel2_size=(3,3), size_pooling=(3,3)):
    modelo = Sequential()
    # Primera capa convolucional
    modelo.add(Conv2D(kernels1, kernel1_size, padding="same", activation="relu", 
                      input_shape=(altura, anchura, canales)))
    modelo.add(MaxPooling2D(pool_size=size_pooling))
    
    # Segunda capa convolucional
    modelo.add(Conv2D(kernels2, kernel2_size, padding="same", activation="relu"))
    modelo.add(MaxPooling2D(pool_size=size_pooling))
    
    # Flatten para conectar a la capa MLP
    modelo.add(Flatten())
    # Primera capa oculta
    modelo.add(Dense(255, activation="relu"))
    # Segunda capa oculta
    modelo.add(Dense(255, activation="relu"))
    # Dropout para evitar overfitting
    modelo.add(Dropout(0.5))
    # Capa de salida
    modelo.add(Dense(clases, activation="softmax"))
    
    modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc","mse"])
    
    return modelo

def entrenar_modelo(ruta_entrenamiento, ruta_validacion,
                     epocas=100, batch_size=2, pasos=100,
                     altura=50, anchura=50):
    # Generar datos para entrenamiento
    datagen_entrenar = ImageDataGenerator(
        rescale=1/255,
        shear_range=0.20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )
    datagen_validar = ImageDataGenerator(rescale=1/255)
    
    imagenes_entrenamiento = datagen_entrenar.flow_from_directory(
        ruta_entrenamiento,
        target_size=(altura,anchura),
        batch_size=batch_size,
        class_mode="categorical"
    )
    
    imagenes_validacion = datagen_validar.flow_from_directory(
        ruta_validacion,
        target_size=(altura,anchura),
        batch_size=batch_size,
        class_mode="categorical"
    )
    
    # Creación del modelo
    modelo = crear_modelo(altura=altura, anchura=anchura, clases=imagenes_entrenamiento.num_classes)
    
    # Callback para guardar el mejor modelo
    checkpoint = ModelCheckpoint("modelo_mejorado.h5", monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    
    # Entrenar el modelo
    historico = modelo.fit(
        imagenes_entrenamiento,
        validation_data=imagenes_validacion,
        epochs=epocas,
        validation_steps=pasos,
        callbacks=[checkpoint],
        verbose=1
    )
    
    return modelo, historico

if __name__ == '__main__':
    # Para pruebas rápidas desde la línea de comandos
    ruta_entrenamiento = os.path.join("..", "datasets", "entrenamiento")
    ruta_validacion = os.path.join("..", "datasets", "validacion")
    modelo, hist = entrenar_modelo(ruta_entrenamiento, ruta_validacion)
