import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Verificar la versión de TensorFlow
print(tf.__version__)

# 1. Carga del dataset Fashion MNIST
mnist = tf.keras.datasets.fashion_mnist

# Dividir el dataset en conjunto de entrenamiento y prueba
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Mostrar las dimensiones de las imágenes y etiquetas de entrenamiento
print(training_images.shape)  # (60000, 28, 28)
print(training_labels.shape)  # (60000,)

# Visualizar algunas imágenes de entrenamiento
print("Primera imagen de la muestra: ")
plt.imshow(training_images[0], cmap=plt.cm.binary)  # Mostrar imagen en escala de grises
plt.title(training_labels[0])
plt.axis('off')  # Ocultar los ejes
plt.show()

# Otra imagen de ejemplo
print("Imagen 50:")
plt.imshow(training_images[50], cmap=plt.cm.binary)
plt.title(training_labels[50])
plt.axis('off')
plt.show()

# Mostrar varias imágenes aleatorias con sus etiquetas
import random
figure, ax = plt.subplots(2, 3)
for i, fila in enumerate(ax):
    for j, col in enumerate(fila):
        ale = random.randint(0, 60000)  # Elegir una imagen aleatoria
        ax[i][j].axis('off')
        ax[i][j].imshow(training_images[ale], cmap=plt.cm.binary)
        ax[i][j].set_title(training_labels[ale])

# 2. Preprocesamiento: Normalización
training_images = training_images / 255  # Escalar valores a [0, 1]
test_images = test_images / 255

# Codificación one-hot de las etiquetas
training_label_hot = tf.one_hot(training_labels, 10)  # Etiquetas de entrenamiento
test_labels_hot = tf.one_hot(test_labels, 10)  # Etiquetas de prueba

# Fijar una semilla para reproducibilidad
tf.random.set_seed(42)

# 3. Definición del modelo
model_zalando = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),  # Capa de entrada (dimensiones de las imágenes)
    tf.keras.layers.Flatten(),  # Aplanar las imágenes
    tf.keras.layers.Dense(512, activation='relu'),  # Primera capa densa
    tf.keras.layers.Dense(512, activation='relu'),  # Segunda capa densa
    tf.keras.layers.Dense(10, activation='softmax')  # Capa de salida para clasificación
])

# Compilación del modelo
model_zalando.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # Optimizador Adam
    loss=tf.keras.losses.CategoricalCrossentropy(),  # Pérdida de entropía cruzada categórica
    metrics=['accuracy']  # Métrica de precisión
)

# Entrenamiento del modelo
history = model_zalando.fit(training_images, training_label_hot, epochs=20)

# Resumen del modelo
model_zalando.summary()

# 4. Evaluación del modelo con el conjunto de prueba
test_loss, test_acc = model_zalando.evaluate(test_images, test_labels_hot)

# Graficar la pérdida y precisión del entrenamiento
pd.DataFrame(history.history).plot()
plt.ylabel("%")
plt.xlabel("Épocas")

# 5. Predicción de una muestra
pred = model_zalando.predict(tf.constant([training_images[20]]))  # Predicción de la imagen 20

# Lista de nombres de clases (traducción de etiquetas a texto)
class_names = ['Camiseta', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']

# Imprimir las clases disponibles
x, idx = tf.unique(training_labels)
for i in x:
    print(class_names[i])

# Imprimir la predicción de la muestra
print(class_names[pred.argmax()])

# 6. Preprocesamiento de imágenes externas
import cv2

def process_image(image_path, change_bg=True, invert_img=False, width=28, height=28):
    """
    Procesa una imagen externa para adaptarla al modelo:
    - Escala de grises
    - Redimensionado a 28x28
    - Opcional: invertir colores y fondo
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"** No se pudo cargar la imagen en la ruta: {image_path} **")
    img = cv2.resize(img, (width, height))
    if change_bg:
        img = cv2.bitwise_not(img)
    if invert_img:
        img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    img = img / 255  # Normalizar a [0, 1]
    return img

# Cargar y procesar una imagen externa
img = process_image('cami.jpg')

# Limpiar cualquier figura anterior
plt.clf()

# Crear una nueva figura para la imagen externa
plt.figure(figsize=(5,5))  # Tamaño de la figura (opcional)
plt.imshow(img, cmap=plt.cm.binary)
plt.axis('off')  # Ocultar los ejes

# Realizar predicción con la imagen procesada
predict = model_zalando.predict(tf.constant([img]))

# Mostrar el título con la predicción
plt.title(f"Predicción: {class_names[predict.argmax()]}")
plt.show()  # Mostrar la imagen

