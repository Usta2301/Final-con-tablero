import numpy as np
import tensorflow as tf
from emnist import extract_training_samples, extract_test_samples
from sklearn.preprocessing import LabelBinarizer
import os

# Crear carpeta para guardar el modelo si no existe
os.makedirs('model', exist_ok=True)

# Cargar datos de EMNIST Balanced
print("Cargando datos EMNIST Balanced...")
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

# Normalizar las imágenes
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Ajustar dimensiones para la CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encoding de etiquetas
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Crear modelo CNN
print("Construyendo modelo...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(47, activation='softmax')  # 47 clases en EMNIST Balanced
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
print("Entrenando modelo...")
model.fit(X_train, y_train, epochs=5, batch_size=128,
          validation_data=(X_test, y_test))

# Guardar el modelo
model_path = 'model/placas_model.h5'
model.save(model_path)
print(f"Modelo guardado en: {model_path}")
