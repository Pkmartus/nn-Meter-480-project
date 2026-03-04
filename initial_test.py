from nn_meter import load_latency_predictor
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Define your architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='leaky_relu'),
    layers.Dense(10, activation='softmax')
])

predictor = load_latency_predictor(predictor_name="cortexA76cpu_tflite21", predictor_version=1.0)

# 3. Predict latency
# Note: Ensure the model is passed as the Keras object and type is 'keras'
latency = predictor.predict(model, model_type='keras')

print(f"Predicted Latency: {latency} ms")