#~Ednoweri Anselmi 26.355.060
import tensorflow as tf
import numpy as np

# Definir los datos de entrada y salida para la compuerta lógica AND
entrada = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
salida = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Definir la arquitectura de la red neuronal
entrada_size = 2
neurona = 4
salida_size = 1

# Definir el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(neurona, activation='relu', input_shape=(entrada_size,)),
    tf.keras.layers.Dense(salida_size, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar la red neuronal
model.fit(entrada, salida, epochs=5000, verbose=0)

# Realizar predicciones
prueba = np.array([[1, 1], [0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float32)
prediccion = model.predict(prueba)

# Mostrar las predicciones
for i in range(len(prueba)):
    print(f'Entrada: {prueba[i]} - Predicción: {prediccion[i][0]}')
