#~Ednoweri Anselmi 26.355.060
from sklearn.neural_network import MLPClassifier

# Definir los datos de entrada y salida para la compuerta lógica AND
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
output_data = [0, 0, 0, 1]

# Crear el clasificador MLP
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', solver='adam', max_iter=5000)

# Entrenar el clasificador
model.fit(input_data, output_data)

# Realizar predicciones
test_data = [[1, 1], [0, 1], [1, 0], [1, 1], [0, 0]]
predictions = model.predict(test_data)

# Mostrar las predicciones
for i in range(len(test_data)):
    print(f'Entrada: {test_data[i]} - Predicción: {predictions[i]}')


    