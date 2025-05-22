import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Cargar el modelo de placas (alfanumérico)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/placas_model.h5")

model = load_model()

def predict_character(image):
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32') / 255.0
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    
    # Mapear índice a carácter
    caracteres = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return caracteres[result] if result < len(caracteres) else "?"

# Interfaz Streamlit
st.set_page_config(page_title='Reconocimiento de caracteres de placa', layout='wide')
st.title('Reconocimiento de caracteres de placas vehiculares')
st.subheader("Dibuja un carácter en el panel y presiona 'Predecir'")

stroke_width = st.slider('Ancho de línea', 1, 30, 15)
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=200,
    width=200,
    key="canvas",
)

if st.button('Predecir'):
    if canvas_result.image_data is not None:
        image = Image.fromarray(np.array(canvas_result.image_data).astype('uint8'), 'RGBA')
        pred = predict_character(image)
        st.header(f'Carácter reconocido: {pred}')
    else:
        st.warning("Por favor dibuja un carácter.")
