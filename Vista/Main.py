# ——————————————————————————————
#             Importaciones
# ——————————————————————————————
import os                             # Manejo de rutas y archivos
import threading                      # Para ejecutar tareas en segundo plano
import cv2                            # Captura y procesamiento de imágenes
import tkinter as tk                  # Interfaz gráfica básica
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw  # Manipulación avanzada de imágenes
from tensorflow.keras.callbacks import Callback  # Para seguimiento de entrenamiento

# Funciones propias del modelo CNN (entrenamiento y predicción)
from Modelo.CnnModelo import entrenar_modelo, predecir_producto  


# ——————————————————————————————
#          Constantes de ruta
# ——————————————————————————————
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# CLASES_DIR: carpeta donde están las imágenes de referencia de cada clase
CLASES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Clases"))


# ——————————————————————————————
#       Variables globales de estado
# ——————————————————————————————
modelo_global      = None    # Guarda el modelo entrenado para usarlo al evaluar
ruta_imagen_subida = ""      # Ruta de la imagen que cargue o capture el usuario


# ——————————————————————————————
#     Información nutricional por clase
# ——————————————————————————————
# Diccionario que asocia nombre de producto → datos a mostrar
info_nutricional = {
    "CocaCola":         "Calorías: 140 kcal\nAzúcares: 39 g\nSodio: 45 mg",
    "Boing":            "Calorías: 110 kcal\nVitaminas: A, C\nFibra: 3 g",
    "JarritoTamarindo": "Calorías: 120 kcal\nAzúcares: 30 g\nSodio: 40 mg\nSabor: Tamarindo"
}

# ——————————————————————————————
#   Imágenes de referencia para cada clase
# ——————————————————————————————
# Se usa al mostrar cuál producto se detectó
default_product_images = {
    "CocaCola":         os.path.join(CLASES_DIR, "coca.png"),
    "Boing":            os.path.join(CLASES_DIR, "boing.png"),
    "JarritoTamarindo": os.path.join(CLASES_DIR, "jarritoTamarindo.png")
}


# ——————————————————————————————
#       Callback para progreso de entrenamiento
# ——————————————————————————————
class ProgressCallback(Callback):
    """
    Se encarga de actualizar la barra y etiqueta de progreso
    cada vez que termina una época de entrenamiento.
    """
    def __init__(self, bar, label, total_epochs):
        self.bar = bar                # widget Progressbar
        self.label = label            # widget Label para porcentaje
        self.total = total_epochs     # total de épocas a entrenar
        self.current = 0              # contador de épocas completadas

    def on_epoch_end(self, epoch, logs=None):
        """
        Se ejecuta automáticamente al finalizar cada época.
        Actualiza la barra según current/total.
        """
        self.current += 1
        self.bar['value'] = self.current
        pct = int(100 * self.current / self.total)
        self.label.config(
            text=f"Entrenando modelo: {pct}% ({self.current}/{self.total} épocas)"
        )
        self.bar.update()


# ——————————————————————————————
#       Funciones auxiliares gráficas
# ——————————————————————————————
def crear_gradiente(width, height, color_start, color_end):
    """
    Genera un degradado vertical de color_start a color_end.
    Parámetros:
      - width, height: dimensiones del degradado.
      - color_start, color_end: tuplas RGB de inicio y fin.
    Devuelve un objeto PhotoImage para Tkinter.
    """
    grad = Image.new("RGB", (width, height), color_start)
    draw = ImageDraw.Draw(grad)
    for y in range(height):
        # Interpolación lineal de cada canal
        r = int(color_start[0] + (color_end[0] - color_start[0]) * y / height)
        g = int(color_start[1] + (color_end[1] - color_start[1]) * y / height)
        b = int(color_start[2] + (color_end[2] - color_start[2]) * y / height)
        draw.line((0, y, width, y), fill=(r, g, b))
    return ImageTk.PhotoImage(grad)


def cargar_imagen(label_widget, path, max_size=(300, 300)):
    """
    Carga una imagen desde archivo, la redimensiona y la muestra
    dentro de un Label de Tkinter con borde decorativo.
    Parámetros:
      - label_widget: widget donde se mostrará la imagen.
      - path: ruta al archivo de imagen.
      - max_size: tamaño máximo (ancho, alto) tras redimensionar.
    """
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen.")
        return
    # Conversión BGR→RGB y a objeto PIL
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail(max_size)  # manteniendo proporciones

    # Borde decorativo de 4 px
    b = 4
    bord = Image.new(
        "RGB",
        (pil.width + 2*b, pil.height + 2*b),
        (41, 128, 185)  # color del borde
    )
    bord.paste(pil, (b, b))

    tkimg = ImageTk.PhotoImage(bord)
    label_widget.config(image=tkimg)
    label_widget.image = tkimg  # referenciar para que no se recolecte


# ——————————————————————————————
#    Funciones de interacción con el usuario
# ——————————————————————————————
def subir_imagen():
    """
    Abre cuadro de diálogo para seleccionar imagen.
    Guarda ruta y la muestra en lbl_upload.
    """
    global ruta_imagen_subida
    fichero = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.jpg;*.png")]
    )
    if fichero:
        ruta_imagen_subida = fichero
        cargar_imagen(lbl_upload, fichero)


def tomar_foto():
    """
    Captura una imagen desde la cámara.
    's' para guardar, 'q' para salir sin guardar.
    Muestra la captura en lbl_upload.
    """
    global ruta_imagen_subida
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a la cámara.")
        return

    messagebox.showinfo(
        "Instrucción",
        "Presiona 's' para capturar, 'q' para salir."
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Captura", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            tmp = os.path.join(os.getcwd(), "temp.jpg")
            cv2.imwrite(tmp, frame)
            ruta_imagen_subida = tmp
            cargar_imagen(lbl_upload, tmp)
            break
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ——————————————————————————————
#       Función para entrenar el modelo
# ——————————————————————————————
def entrenar():
    """
    Inicia el proceso de entrenamiento en un hilo aparte
    y actualiza la interfaz con barra de progreso.
    """
    global modelo_global
    # Configurar barra y etiqueta
    progress_bar['maximum'] = EPOCHS
    progress_bar['value'] = 0
    progress_bar.place(x=20, y=580, width=600, height=20)
    progress_label.place(x=630, y=575)
    mensaje.config(text="⏳ Entrenando modelo...", fg="white")

    def job():
        # Rutas a los datos de entrenamiento y validación
        ruta_train = os.path.join(BASE_DIR, "..", "Datasets", "entrenamiento")
        ruta_val   = os.path.join(BASE_DIR, "..", "Datasets", "validacion")
        # Callback para actualizar progreso
        prog = ProgressCallback(progress_bar, progress_label, total_epochs=EPOCHS)

        # Llamada a la función de entrenamiento real
        modelo, _ = entrenar_modelo(
            ruta_train, ruta_val,
            epocas=EPOCHS,
            batch_size=BATCH_SIZE,
            pasos_val=STEPS_VAL,
            altura=IMG_H,
            anchura=IMG_W,
            extra_callbacks=[prog]
        )

        # Guardar modelo entrenado y ocultar indicadores
        globals()['modelo_global'] = modelo
        progress_bar.place_forget()
        progress_label.place_forget()
        mensaje.config(text="✅ Modelo entrenado correctamente", fg="lightgreen")

    # Ejecutar en hilo para no bloquear la GUI
    threading.Thread(target=job, daemon=True).start()


# ——————————————————————————————
#      Función para evaluar una imagen
# ——————————————————————————————
def evaluar():
    """
    Usa el modelo entrenado para predecir
    y muestra el resultado con su imagen e info.
    """
    if modelo_global is None:
        messagebox.showerror("Error", "Entrena primero el modelo.")
        return
    if not ruta_imagen_subida:
        messagebox.showerror("Error", "Sube o toma una imagen primero.")
        return

    # Predicción con umbral de confianza
    pred = predecir_producto(ruta_imagen_subida, modelo_global, umbral=UMBRAL)

    # Mensaje según pertenezca o no al catálogo
    if pred in info_nutricional:
        messagebox.showinfo("⚠️ Producto detectado", f"Bebida dañina: {pred}")
    else:
        messagebox.showwarning("❓ No identificado", "Producto no identificado en el catálogo.")

    # Seleccionar imagen de referencia según la predicción
    if pred in default_product_images:
        ref = default_product_images[pred]
    else:
        ref = os.path.join(CLASES_DIR, "desc.jpg")

    # Mostrar imagen y texto de resultado
    if os.path.exists(ref):
        cargar_imagen(lbl_result_img, ref, max_size=(220, 220))
    lbl_result_text.config(
        text=f"{pred}\n\n{info_nutricional.get(pred, 'Sin información.')}"
    )


# ——————————————————————————————
#    Parámetros ajustables de entrenamiento
# ——————————————————————————————
EPOCHS     = 10    # Número de ciclos completos sobre todo el dataset
BATCH_SIZE = 16    # Cantidad de imágenes procesadas por paso
STEPS_VAL  = 50    # Pasos de validación en cada época
IMG_H, IMG_W = 100, 100  # Tamaño al que se redimensionan las imágenes
UMBRAL     = 0.6   # Nivel mínimo de confianza para considerar válida la predicción


# ——————————————————————————————
#          Construcción de la GUI
# ——————————————————————————————
vent = tk.Tk()
vent.title("NutriScan")
vent.geometry("1000x730")
vent.configure(bg="#2c3e50")
vent.resizable(False, False)

# Estilos personalizados para widgets
style = ttk.Style(vent)
style.theme_use("clam")
style.configure("Custom.Horizontal.TProgressbar",
                troughcolor="#34495e", background="#1abc9c")
style.configure("TButton",
                font=("Segoe UI", 11, "bold"),
                relief="flat", borderwidth=0, padding=8)
style.map("TButton",
          background=[("!active", "#2980b9"), ("active", "#3498db")],
          foreground=[("!active", "white"), ("active", "white")])
style.configure("TLabel",
                background="#2c3e50", foreground="white",
                font=("Segoe UI", 10))
style.configure("Header.TLabel",
                font=("Segoe UI", 22, "bold"), foreground="white")

# Header con degradado
vent.grad_head = crear_gradiente(1000, 80, (44, 62, 80), (52, 152, 219))
hdr = ttk.Label(vent, image=vent.grad_head)
hdr.image = vent.grad_head
hdr.place(x=0, y=0, width=1000, height=80)
ttk.Label(hdr, text="NutriScan", style="Header.TLabel")\
   .place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Panel principal
main = tk.Frame(vent, bg="#2c3e50")
main.place(x=20, y=100, width=960, height=500)

# Área de imagen cargada
lf = tk.LabelFrame(main, text="Imagen subida",
                   bg="#34495e", fg="white",
                   font=("Segoe UI", 12, "bold"),
                   bd=2, relief="ridge", labelanchor="n")
lf.place(x=20, y=20, width=420, height=460)
lbl_upload = tk.Label(lf, bg="#34495e")
lbl_upload.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Área de resultado
rf = tk.LabelFrame(main, text="Resultado del análisis",
                   bg="#34495e", fg="white",
                   font=("Segoe UI", 12, "bold"),
                   bd=2, relief="ridge", labelanchor="n")
rf.place(x=460, y=20, width=480, height=460)
lbl_result_img = tk.Label(rf, bg="#34495e")
lbl_result_img.place(relx=0.5, rely=0.25, anchor=tk.CENTER,
                     width=220, height=220)
lbl_result_text = tk.Label(
    rf, text="Aquí aparecerá el resultado",
    bg="#34495e", fg="white",
    font=("Segoe UI", 12),
    justify="center", wraplength=420
)
lbl_result_text.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

# Barra de progreso (invisible hasta entrenar)
progress_bar   = ttk.Progressbar(
    vent, mode='determinate', style="Custom.Horizontal.TProgressbar"
)
progress_label = ttk.Label(
    vent, text="", background="#2c3e50",
    foreground="white", font=("Segoe UI", 10)
)

# Botonera inferior
botf = tk.Frame(vent, bg="#2c3e50")
botf.place(x=20, y=620, width=960, height=60)
ttk.Button(botf, text="🖼️ Subir Imagen",   command=subir_imagen).grid(row=0, column=0, padx=10)
ttk.Button(botf, text="📸 Tomar Foto",      command=tomar_foto ).grid(row=0, column=1, padx=10)
ttk.Button(botf, text="⚙️ Entrenar Modelo", command=entrenar  ).grid(row=0, column=2, padx=10)
ttk.Button(botf, text="🔍 Evaluar Imagen",  command=evaluar   ).grid(row=0, column=3, padx=10)

# Mensaje de estado en la parte inferior
mensaje = tk.Label(
    vent, text="Sistema listo",
    bg="#2c3e50", fg="white",
    font=("Segoe UI", 11)
)
mensaje.place(x=20, y=690, width=960, height=30)

# Iniciar bucle de eventos de la ventana
vent.mainloop()
