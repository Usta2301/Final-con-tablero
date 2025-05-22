import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os

# ======= CONFIGURACIÓN =======
MODEL_PATH = "model/placas_model.h5"

# 47 clases del dataset EMNIST Balanced en orden específico
CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

# ======= CARGA DEL MODELO =======
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo en {MODEL_PATH}. Entrénalo primero con 'train_placas_model.py'")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======= FUNCIÓN DE PREDICCIÓN =======
def predict_character(image):
    # Convertir a escala de grises
    image = ImageOps.grayscale(image)
    # Redimensionar a 28x28
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return CARACTERES[result] if result < len(CARACTERES) else "?"

# ======= INTERFAZ STREAMLIT =======
st.set_page_config(page_title='Reconocimiento de caracteres de placas', layout='wide')
st.title('🧠 Reconocimiento de caracteres de placas vehiculares')
st.subheader("✍️ Dibuja un carácter (letra o número) y presiona 'Predecir'")

# Parámetros del canvas
stroke_width = st.slider('🖌️ Ancho de línea', 1, 30, 15)
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # color de fondo del trazo
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',  # color del trazo
    background_color='#000000',  # fondo negro
    height=200,
    width=200,
    key="canvas",
)

# Botón para predecir
if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        image = Image.fromarray(np.array(canvas_result.image_data).astype('uint8'), 'RGBA')
        pred = predict_character(image)
        st.success(f"✅ Carácter reconocido: **{pred}**")
    else:
        st.warning("Por favor dibuja un carácter sobre el lienzo.")

# Sidebar
st.sidebar.title("ℹ️ Acerca de")
st.sidebar.info("""
Esta app permite reconocer caracteres alfanuméricos escritos a mano, 
como los que aparecen en placas vehiculares.

Modelo entrenado con EMNIST Balanced.
""")
