# Clasificador de Imágenes con TensorFlow y Fashion MNIST

Este proyecto utiliza TensorFlow para entrenar un modelo de clasificación de imágenes sobre el conjunto de datos Fashion MNIST, que contiene imágenes de ropa, como camisetas, pantalones, vestidos, entre otros. El modelo se entrena, evalúa y se usa para hacer predicciones sobre imágenes tanto del conjunto de prueba como imágenes externas.

## Requisitos

Para ejecutar este proyecto, necesitas tener instalados los siguientes paquetes de Python:

- `tensorflow` (para la creación y entrenamiento del modelo)
- `numpy` (para manejar arrays y datos numéricos)
- `matplotlib` (para la visualización de resultados)
- `pandas` (para manejar los resultados y graficar la precisión del modelo)
- `opencv-python` (para procesar imágenes externas)

Puedes instalar las dependencias con:

```bash
pip install tensorflow numpy matplotlib pandas opencv-python
```
## Estructura del Proyecto

El proyecto consta de los siguientes pasos principales:

### Carga del dataset Fashion MNIST:
- Se carga el conjunto de datos de imágenes de ropa (60,000 imágenes de entrenamiento y 10,000 imágenes de prueba).
- Se visualizan algunas imágenes y sus etiquetas.

### Preprocesamiento de los datos:
- Las imágenes se normalizan para que los valores de los píxeles estén en el rango [0, 1].
- Las etiquetas se codifican en formato one-hot.

### Creación del modelo de clasificación:
- El modelo es una red neuronal con dos capas densas (512 neuronas cada una) y una capa de salida con 10 neuronas, que corresponde a las 10 clases posibles.
- Se usa el optimizador Adam y la función de pérdida CategoricalCrossentropy.

### Entrenamiento del modelo:
- El modelo se entrena durante 20 épocas usando el conjunto de entrenamiento.

### Evaluación del modelo:
- Se evalúa la precisión del modelo en el conjunto de prueba.
- Se muestra un gráfico de la precisión y la pérdida durante el entrenamiento.

### Predicciones:
- El modelo hace predicciones sobre una imagen específica del conjunto de entrenamiento.
- Las clases se traducen de números a nombres de prendas de ropa.

### Predicción con imágenes externas:
- Se carga y procesa una imagen externa para hacer predicciones con el modelo entrenado.
- La imagen es redimensionada, convertida a escala de grises y normalizada antes de realizar la predicción.

## Cómo Usar el Proyecto

### Entrenamiento del modelo:
- Ejecuta el código en tu entorno local o en un entorno Jupyter Notebook.
- Asegúrate de que las dependencias estén instaladas y que tengas acceso a TensorFlow.

### Predicción de una imagen externa:
- Coloca una imagen de ropa (por ejemplo, una camiseta) en el directorio de tu proyecto y cambia el nombre del archivo en la función `process_image` para que coincida con la ruta de la imagen que deseas procesar.
- La imagen se procesará, y el modelo hará una predicción.

## Ejemplo de Imagen de Entrada

El modelo toma una imagen de 28x28 píxeles, en escala de grises, que representa una prenda de ropa. A continuación, se muestra un ejemplo de cómo el modelo puede predecir una imagen externa (por ejemplo, una camiseta):

```python
img = process_image('cami.jpg')
predict = model_zalando.predict(tf.constant([img]))
plt.title(f"Predicción: {class_names[predict.argmax()]}")
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
```
## Resultados del Modelo

Después de entrenar el modelo, se mostrará un gráfico con la precisión y la pérdida a lo largo de las épocas.

La predicción para una muestra del conjunto de datos y para una imagen externa se mostrará con su respectiva etiqueta.

## Contribuciones

Si deseas contribuir a este proyecto, no dudes en hacer un fork y enviar un pull request con tus mejoras o correcciones.
