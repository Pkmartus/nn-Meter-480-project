import tf2onnx
from keras import models, Input, layers
import tensorflow as tf
from nn_meter import load_latency_predictor
import warnings
import logging

# Mute scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Mute nn-Meter info logs and version warnings
logging.getLogger('nn_meter').setLevel(logging.ERROR)

# Define CNN model architecture
model = models.Sequential([
    Input((32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
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

# Convert TensorFlow Keras model to ONNX format
# (use a batch size of 1 to simulate single-image edge inference)
input_signature = [tf.TensorSpec([1, 32, 32, 3], tf.float32, name='x')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

"""
TODO: Change functionality to allow users to enter their own custom 
model filename for use instead of current hardcoded method. 

TODO: Create function that maps model file extension to variable for 
model_type parameter used by nnMeterPredictor.

TODO: Add loop to allow the user to test another model or a different
inference framework if they want to.
"""

# Save the ONNX file (write binary)
model_filename = 'custom_model.onnx'
with open(model_filename, 'wb') as f:
    f.write(onnx_model.SerializeToString())

inference_frameworks = ['cortexA76cpu_tflite21', 'adreno640gpu_tflite21',
                        'adreno630gpu_tflite21', 'myriadvpu_openvino2019r2']
num_frameworks = len(inference_frameworks)

while (True):
    print(f"Using model: {model_filename}\n")
    print('Options:')
    for i, platform in enumerate(inference_frameworks, start=1):
        print(f"{i}. {platform}")

    # Prompt user for input
    user_input = input(f"\nEnter a number (1-{num_frameworks}) to "
                       "select a device inference framework to perform "
                       "latency prediction on: "
                       )

    try:
        # Check if the user entered a non-numeric value or invalid option
        if (not user_input.isnumeric):
            raise ValueError
        user_input = int(user_input)  # typecast input to integer
        if (user_input < 1 or user_input > num_frameworks):
            raise ValueError
        else:
            break  # valid input = exit the loop
    except:
        print("Error: Invalid option.\n")
        continue

# Predict the inference latency of the model on the device
predictor = load_latency_predictor(
    predictor_name=inference_frameworks[user_input - 1], predictor_version=1.0)

latency = predictor.predict('custom_model.onnx', model_type='onnx')
