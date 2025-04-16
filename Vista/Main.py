# gui/main.py
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import threading
from tensorflow import keras

# Importar la función de entrenamiento y carga del modelo
from Modelo import CnnModelo  # Asegúrate de que CnnModelo tenga la función entrenar_modelo()

# Variables globales
modelo_global = None
ruta_imagen_subida = ""
imagen_actual = None

# Diccionario con datos nutricionales de ejemplo
info_nutricional = {
    "CocaCola": "Información nutricional de CocaCola:\n- Calorías: 140 kcal\n- Azúcares: 39 g\n- Sodio: 45 mg",
    "Boing": "Información nutricional de Boing:\n- Calorías: 110 kcal\n- Vitaminas: A, C\n- Fibra: 3 g",
    "Otra": "Información nutricional del producto:\n- Datos no disponibles.\n- Consulte la etiqueta."
}

def crear_gradiente(width, height, color1, color2):
    gradient = Image.new('RGB', (width, height), color1)
    draw = ImageDraw.Draw(gradient)
    for y in range(height):
        r = int(color1[0] + (color2[0] - color1[0]) * y / height)
        g = int(color1[1] + (color2[1] - color1[1]) * y / height)
        b = int(color1[2] + (color2[2] - color1[2]) * y / height)
        draw.line((0, y, width, y), fill=(r, g, b))
    return ImageTk.PhotoImage(gradient)

def cargar_imagen(label, path, max_size=(300, 300)):
    global imagen_actual
    img_cv = cv2.imread(path)
    if img_cv is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen.")
        return
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv)
    img_pil.thumbnail(max_size)

    # Agregar borde decorativo
    border_size = 3
    img_with_border = Image.new("RGB", 
                                (img_pil.width + border_size*2, img_pil.height + border_size*2),
                                (41, 128, 185))
    img_with_border.paste(img_pil, (border_size, border_size))
    
    img_tk = ImageTk.PhotoImage(img_with_border)
    label.config(image=img_tk)
    label.image = img_tk  # Conservar referencia
    imagen_actual = img_with_border  # Guardar la imagen actual (opcional)

def subir_imagen():
    global ruta_imagen_subida
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")]
    )
    if ruta:
        ruta_imagen_subida = ruta
        cargar_imagen(lbl_imagen, ruta)
        
        producto_detectado = predecir_producto(ruta)
        combo_producto.set(producto_detectado)  # Auto-selecciona en el combobox
        actualizar_info_nutricional()

def tomar_foto():
    global ruta_imagen_subida
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a la cámara.")
        return

    messagebox.showinfo("Tomar foto", "Presione la tecla 's' para capturar la imagen y 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo capturar la imagen.")
            break
        cv2.imshow("Presione 's' para capturar, 'q' para salir", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            ruta_temp = os.path.join(os.getcwd(), "temp_captura.jpg")
            cv2.imwrite(ruta_temp, frame)
            ruta_imagen_subida = ruta_temp
            cargar_imagen(lbl_imagen, ruta_temp)
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

    producto_detectado = predecir_producto(ruta_imagen_subida)
    combo_producto.set(producto_detectado)
    actualizar_info_nutricional()


def entrenar():
    global modelo_global
    mensaje.config(text="Entrenando modelo...", foreground="white", image=gradiente_msg, compound=tk.CENTER)
    def run_entrenamiento():
        ruta_entrenamiento = os.path.join("..", "datasets", "entrenamiento")
        ruta_validacion = os.path.join("..", "datasets", "validacion")
        modelo, hist = CnnModelo.entrenar_modelo(ruta_entrenamiento, ruta_validacion, epocas=10)
        global modelo_global
        modelo_global = modelo
        mensaje.config(text="✅ Entrenamiento completado!", image="", foreground="white")
    threading.Thread(target=run_entrenamiento).start()
    
def predecir_producto(ruta_imagen):
    global modelo_global
    if modelo_global is None:
        messagebox.showwarning("Modelo no entrenado", "Por favor, entrena el modelo primero.")
        return "Otra"

    # Procesar imagen como fue entrenada
    img = keras.utils.load_img(ruta_imagen, target_size=(150, 150))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Hacer predicción
    prediccion = modelo_global.predict(img_array)
    clase_predicha = np.argmax(prediccion)

    # Mapea el índice a etiqueta (ajusta esto con tus etiquetas reales)
    etiquetas = ["CocaCola", "Boing", "Otra"]  # Asegúrate que este orden coincida con el del entrenamiento
    return etiquetas[clase_predicha]

def analizar_producto():
    if not ruta_imagen_subida:
        messagebox.showwarning("Imagen no cargada", "Por favor, sube una imagen o toma una foto primero.")
        return

    producto_detectado = predecir_producto(ruta_imagen_subida)
    combo_producto.set(producto_detectado)
    actualizar_info_nutricional()
    messagebox.showinfo("Análisis completado", f"Producto detectado: {producto_detectado}")



def actualizar_info_nutricional(*args):
    # Obtener el producto seleccionado desde el combobox
    producto = combo_producto.get()
    info = info_nutricional.get(producto, "Información no disponible.")
    lbl_info.config(text=info)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Clasificador de Productos Saludables")
ventana.geometry("1000x700")
ventana.configure(bg="#2c3e50")
ventana.resizable(False, False)

# Configurar estilos con ttk para una apariencia moderna
style = ttk.Style(ventana)
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 10, "bold"), borderwidth=0, relief="flat")
style.map("TButton",
    foreground=[('active', 'white'), ('!active', 'white')],
    background=[('active', '#3498db'), ('!active', '#2980b9')])
style.configure("TLabel", font=("Segoe UI", 10), background="#2c3e50", foreground="white")
style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), background="#2c3e50", foreground="white")

# Crear gradientes para cabecera, botones y mensajes
gradient_header = crear_gradiente(1000, 80, (44, 62, 80), (52, 152, 219))
gradient_btn = crear_gradiente(150, 40, (41, 128, 185), (52, 152, 219))
gradiente_msg = crear_gradiente(800, 30, (46, 134, 193), (52, 152, 219))

# Cabecera
header = ttk.Label(ventana, image=gradient_header)
header.place(x=0, y=0, width=1000, height=80)
ttk.Label(header, text="Comida Saludable", style="Header.TLabel").place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Frame principal para el contenido
main_frame = ttk.Frame(ventana, style="TLabel")
main_frame.place(x=20, y=100, width=960, height=500)

# Panel izquierdo: Área de previsualización de imagen
img_frame = ttk.Frame(main_frame, borderwidth=3, relief="ridge")
img_frame.place(x=30, y=20, width=400, height=400)
lbl_imagen = ttk.Label(img_frame, background="#34495e")
lbl_imagen.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Panel derecho: Controles y selección de producto
control_frame = ttk.Frame(main_frame, style="TLabel")
control_frame.place(x=450, y=20, width=480, height=150)
# Botón "Analizar Producto"



# Botón "Subir Imagen"
btn_subir = ttk.Button(control_frame, text="🖼️ Subir Imagen", command=subir_imagen)
btn_subir.grid(row=0, column=0, padx=20, pady=10, ipadx=10, ipady=5)

# Botón "Tomar Foto"
btn_tomar = ttk.Button(control_frame, text="📸 Tomar Foto", command=tomar_foto)
btn_tomar.grid(row=0, column=1, padx=20, pady=10, ipadx=10, ipady=5)

# Botón "Entrenar Modelo" (solo actualiza el entrenamiento sin que el usuario suba fotos)
btn_entrenar = ttk.Button(control_frame, text="⚙️ Entrenar Modelo", command=entrenar)
btn_entrenar.grid(row=1, column=0, columnspan=2, padx=10, pady=10, ipadx=10, ipady=5)

btn_analizar = ttk.Button(control_frame, text="⚙️Analizar", command=analizar_producto)
btn_analizar.grid(row=1, column=1, columnspan=2, padx=10, pady=10, ipadx=10, ipady=5)

# Desplegable para seleccionar el tipo de producto
ttk.Label(control_frame, text="Selecciona Tipo de Producto:", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, padx=20, pady=5, sticky=tk.W)
combo_producto = ttk.Combobox(control_frame, values=["CocaCola", "Boing", "Otra"], state="readonly", font=("Segoe UI", 10))
combo_producto.current(0)
combo_producto.grid(row=2, column=1, padx=20, pady=5)
combo_producto.bind("<<ComboboxSelected>>", actualizar_info_nutricional)

# Panel inferior: Mostrar información del producto seleccionado (datos nutricionales, etc.)
status_frame = ttk.Frame(main_frame, borderwidth=3, relief="ridge")
status_frame.place(x=30, y=440, width=900, height=50)
lbl_info = ttk.Label(status_frame, text="Información nutricional aquí", font=("Segoe UI", 11), background="#34495e", foreground="white", anchor="center")
lbl_info.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Mensaje de estado (ubicado sobre la ventana inferior)
mensaje = ttk.Label(ventana, text="Sistema listo", font=("Segoe UI", 11), background="#2c3e50", foreground="white")
mensaje.place(x=20, y=620, width=960, height=50)

ventana.mainloop()
