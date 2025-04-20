import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import threading

from Modelo.CnnModelo import entrenar_modelo, predecir_producto

# — Constantes de ruta —
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLASES_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "Clases"))

# — Variables globales —
modelo_global      = None
ruta_imagen_subida = ""

# — Info nutricional corregida —
info_nutricional = {
    "CocaCola":         "Calorías: 140 kcal\nAzúcares: 39 g\nSodio: 45 mg",
    "Boing":            "Calorías: 110 kcal\nVitaminas: A, C\nFibra: 3 g",
    "JarritoTamarindo": "Calorías: 120 kcal\nAzúcares: 30 g\nSodio: 40 mg\nSabor: Tamarindo"
}

# — PNGs de cada clase en Clases/ —
default_product_images = {
    "CocaCola":         os.path.join(CLASES_DIR, "coca.png"),
    "Boing":            os.path.join(CLASES_DIR, "boing.png"),
    "JarritoTamarindo": os.path.join(CLASES_DIR, "jarritoTamarindo.png")
}

def crear_gradiente(w, h, c1, c2):
    grad = Image.new("RGB", (w, h), c1)
    draw = ImageDraw.Draw(grad)
    for y in range(h):
        r = int(c1[0] + (c2[0] - c1[0]) * y / h)
        g = int(c1[1] + (c2[1] - c1[1]) * y / h)
        b = int(c1[2] + (c2[2] - c1[2]) * y / h)
        draw.line((0, y, w, y), fill=(r, g, b))
    return ImageTk.PhotoImage(grad)

def cargar_imagen(label, path, max_size=(300, 300)):
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen.")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail(max_size)
    b = 4
    bord = Image.new("RGB", (pil.width + 2*b, pil.height + 2*b), (41, 128, 185))
    bord.paste(pil, (b, b))
    tkimg = ImageTk.PhotoImage(bord)
    label.config(image=tkimg)
    label.image = tkimg

def subir_imagen():
    global ruta_imagen_subida
    f = filedialog.askopenfilename(
        title="Selecciona imagen", filetypes=[("Imágenes", "*.jpg;*.png")]
    )
    if f:
        ruta_imagen_subida = f
        cargar_imagen(lbl_upload, f)

def tomar_foto():
    global ruta_imagen_subida
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a la cámara.")
        return
    messagebox.showinfo("Instrucción", "Presiona 's' para capturar, 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret: break
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

def entrenar():
    global modelo_global
    mensaje.config(text="", image=grad_msg, compound=tk.CENTER)
    def job():
        ent = os.path.join(BASE_DIR, "..", "Datasets", "entrenamiento")
        val = os.path.join(BASE_DIR, "..", "Datasets", "validacion")
        modelo, _ = entrenar_modelo(ent, val, epocas=10)
        globals()['modelo_global'] = modelo
        mensaje.config(text="✅ Entrenamiento completado", image="", foreground="white")
    threading.Thread(target=job, daemon=True).start()

def evaluar():
    if modelo_global is None:
        messagebox.showerror("Error", "Entrena primero.")
        return
    if not ruta_imagen_subida:
        messagebox.showerror("Error", "Sube o toma imagen primero.")
        return

    pred = predecir_producto(ruta_imagen_subida, modelo_global)

    # Mensaje según predicción
    if pred in info_nutricional:
        messagebox.showinfo(
            "⚠️ Producto detectado",
            f"Bebida dañina para la salud.\n\nA continuación la info de: {pred}"
        )
    else:
        messagebox.showwarning(
            "❓ No identificado",
            "Producto no identificado en nuestro catálogo de productos dañinos."
        )

    # Mostrar imagen de resultado
    ref = default_product_images.get(pred, "")
    if os.path.exists(ref):
        cargar_imagen(lbl_result_img, ref, max_size=(220, 220))
    else:
        lbl_result_img.config(image='')
        lbl_result_img.image = None

    # Mostrar texto de resultado
    lbl_result_text.config(
        text=f"{pred}\n\n{info_nutricional.get(pred, 'Sin información disponible.')}"
    )

    # Efecto de resaltado del panel de resultado
    original_bg = rf.cget("bg")
    rf.config(bg="#16a085")
    vent.after(200, lambda: rf.config(bg=original_bg))

    mensaje.config(text="✅ Imagen evaluada", image="", foreground="white")

# — Construcción de la ventana —
vent = tk.Tk()
vent.title("NutriScan")
vent.geometry("1000x700")
vent.configure(bg="#2c3e50")
vent.resizable(False, False)

# — Estilos ttk —
style = ttk.Style(vent)
style.theme_use("clam")
style.configure(
    "TButton",
    font=("Segoe UI", 11, "bold"),
    relief="flat",
    borderwidth=0,
    padding=8
)
style.map("TButton",
    background=[("!active", "#2980b9"), ("active", "#3498db")],
    foreground=[("!active", "white"), ("active", "white")]
)
style.configure("TLabel", background="#2c3e50", foreground="white", font=("Segoe UI", 10))
style.configure("Header.TLabel", font=("Segoe UI", 22, "bold"), foreground="white")

# — Gradientes —
grad_head = crear_gradiente(1000, 80, (44, 62, 80), (52, 152, 219))
grad_msg  = crear_gradiente(800, 30, (46, 134, 193), (52, 152, 219))

# — Header —
hdr = ttk.Label(vent, image=grad_head)
hdr.place(x=0, y=0, width=1000, height=80)
ttk.Label(
    hdr,
    text="NutriScan",
    style="Header.TLabel"
).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# — Panel principal —
main = tk.Frame(vent, bg="#2c3e50")
main.place(x=20, y=100, width=960, height=500)

# — Panel izquierda: Imagen subida —
lf = tk.LabelFrame(
    main,
    text="Imagen subida",
    bg="#34495e",
    fg="white",
    font=("Segoe UI", 12, "bold"),
    bd=2,
    relief="ridge",
    labelanchor="n"
)
lf.place(x=20, y=20, width=420, height=460)
lbl_upload = tk.Label(lf, bg="#34495e")
lbl_upload.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# — Panel derecha: Resultado —
rf = tk.LabelFrame(
    main,
    text="Resultado del análisis",
    bg="#34495e",
    fg="white",
    font=("Segoe UI", 12, "bold"),
    bd=2,
    relief="ridge",
    labelanchor="n"
)
rf.place(x=460, y=20, width=480, height=460)

lbl_result_img = tk.Label(rf, bg="#34495e")
lbl_result_img.place(relx=0.5, rely=0.25, anchor=tk.CENTER, width=220, height=220)

lbl_result_text = tk.Label(
    rf,
    text="Aquí aparecerá el resultado",
    bg="#34495e",
    fg="white",
    font=("Segoe UI", 12),
    justify="center",
    wraplength=420
)
lbl_result_text.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

# — Botonera —
botf = tk.Frame(vent, bg="#2c3e50")
botf.place(x=20, y=620, width=960, height=60)

ttk.Button(botf, text="🖼️ Subir Imagen", command=subir_imagen).grid(row=0, column=0, padx=10)
ttk.Button(botf, text="📸 Tomar Foto",    command=tomar_foto).grid(  row=0, column=1, padx=10)
ttk.Button(botf, text="⚙️ Entrenar Modelo", command=entrenar).grid( row=0, column=2, padx=10)
ttk.Button(botf, text="🔍 Evaluar Imagen",  command=evaluar).grid(  row=0, column=3, padx=10)

mensaje = tk.Label(
    vent,
    text="Sistema listo",
    bg="#2c3e50",
    fg="white",
    font=("Segoe UI", 11)
)
mensaje.place(x=20, y=690, width=960, height=30)

vent.mainloop()
