import tf2onnx
from keras import models, Input, layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
from nn_meter import load_latency_predictor
import warnings
import logging

"""
TODO: Change functionality to allow users to enter their own custom
model filename for use instead of current hardcoded method.

TODO: Create function that maps model file extension to variable for
model_type parameter used by nnMeterPredictor.
"""

# Mute scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Mute nn-Meter info logs and version warnings
logging.getLogger('nn_meter').setLevel(logging.ERROR)

# Define hardware and inference frameworks supported by nn-Meter
hardware_inference_frameworks = [
    'cortexA76cpu_tflite21',
    'adreno640gpu_tflite21',
    'adreno630gpu_tflite21',
    'myriadvpu_openvino2019r2'
]
num_frameworks = len(hardware_inference_frameworks)

def createVGGNet():
    model = VGG19(weights='imagenet')
    return model

def createResNet():
    model = ResNet50(weights='imagenet')
    return model

def createAlexNet():
    #source: https://github.com/animikhaich/AlexNet-Tensorflow/blob/main/AlexNet_Prototype_Model.ipynb?short_path=ff15766
    # Create AlexNet Model (Vanilla)

    # Input Layer
    inputs = tf.keras.Input(shape=(224, 224, 3), name="alexnet_input")

    # Layer 1 - Convolutions
    l1_g1 = tf.keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, padding="same")(inputs)
    l1_g1 = tf.keras.layers.BatchNormalization()(l1_g1)
    l1_g1 = tf.keras.layers.ReLU()(l1_g1)
    l1_g1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l1_g1)

    l1_g2 = tf.keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, padding="same")(inputs)
    l1_g2 = tf.keras.layers.BatchNormalization()(l1_g2)
    l1_g2 = tf.keras.layers.ReLU()(l1_g2)
    l1_g2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l1_g2)

    # Layer 2 - Convolutions
    l2_g1 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same")(l1_g1)
    l2_g1 = tf.keras.layers.BatchNormalization()(l2_g1)
    l2_g1 = tf.keras.layers.ReLU()(l2_g1)
    l2_g1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l2_g1)

    l2_g2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, padding="same")(l1_g2)
    l2_g2 = tf.keras.layers.BatchNormalization()(l2_g2)
    l2_g2 = tf.keras.layers.ReLU()(l2_g2)
    l2_g2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l2_g2)

    # Layer 3 - Convolutions
    l3_concat = tf.keras.layers.concatenate([l2_g1, l2_g2], axis=-1)

    l3_g1 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same")(l3_concat)
    l3_g1 = tf.keras.layers.ReLU()(l3_g1)

    l3_g2 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same")(l3_concat)
    l3_g2 = tf.keras.layers.ReLU()(l3_g2)

    # Layer 4 - Convolutions
    l4_g1 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same")(l3_g1)
    l4_g1 = tf.keras.layers.ReLU()(l4_g1)

    l4_g2 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding="same")(l3_g2)
    l4_g2 = tf.keras.layers.ReLU()(l4_g2)

    # Layer 5 - Convolutions
    l5_g1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(l4_g1)
    l5_g1 = tf.keras.layers.ReLU()(l5_g1)
    l5_g1 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l5_g1)

    l5_g2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(l4_g2)
    l5_g2 = tf.keras.layers.ReLU()(l5_g2)
    l5_g2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)(l5_g2)

    # Layer 6 - Dense
    l6_pre = tf.keras.layers.concatenate([l5_g1, l5_g2], axis=-1)
    l6_pre = tf.keras.layers.Flatten()(l6_pre)

    l6 = tf.keras.layers.Dense(units=4096)(l6_pre)
    l6 = tf.keras.layers.ReLU()(l6)
    l6 = tf.keras.layers.Dropout(rate=0.5)(l6)

    # Layer 7 - Dense
    l7 = tf.keras.layers.Dense(units=4096)(l6)
    l7 = tf.keras.layers.ReLU()(l7)
    l7 = tf.keras.layers.Dropout(rate=0.5)(l7)

    # Layer 8 - Dense
    l8 = tf.keras.layers.Dense(units=1000)(l7)
    l8 = tf.keras.layers.Softmax(dtype=tf.float32, name="alexnet_output")(l8)

    alexnet = tf.keras.models.Model(inputs=inputs, outputs=l8)
    return alexnet



def createModel():
    # Define CNN model architecture
    return models.Sequential([
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


def convertToONNX(model, filename):
    # Convert TensorFlow Keras model to ONNX format
    # Dynamically get the input shape from the model
    # model.input_shape returns (None, H, W, C), so we replace None with 1 for batch size
    input_shape = list(model.input_shape)
    input_shape[0] = 1 
    
    # Create the signature based on the actual model requirements
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature, opset=13)

    # Save the ONNX file (write binary)
    with open(filename, 'wb') as f:
        f.write(onnx_model.SerializeToString())


def main():
    # Define CNN model architecture
    model = createModel()

    # Convert model to onnx format
    model_filename = 'custom_model.onnx'
    convertToONNX(model, model_filename)

    # Handle user input
    while (True):
        while (True):
            print(f"Using model: {model_filename}\n")
            print('Options:')
            for i, platform in enumerate(hardware_inference_frameworks, start=1):
                print(f"{i}. {platform}")

            # Prompt user for input
            user_input = input(
                f"\nEnter a number (1-{num_frameworks}) to "
                "select a device inference framework to perform "
                "latency prediction on: "
            )

            try:
                # Check if user entered a non-numeric value or invalid option
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
            predictor_name=hardware_inference_frameworks[user_input - 1],
            predictor_version=1.0
        )

        predictor.predict('custom_model.onnx', model_type='onnx')
        print("\n", end="")  # print a newline afterwards

        while (True):
            # Prompt user to predict latency for another framework
            user_input = input(
                "Would you like to predict the latency of another hardware "
                "and inference framework? Enter y/n: "
            )
            try:
                if (user_input.isnumeric()):
                    raise ValueError
                # Convert user input to lowercase for comparison
                user_input = user_input.strip().lower()
                if (user_input != "y" and user_input != "n"):
                    raise ValueError
                elif (user_input == "y"):
                    continue
                elif (user_input == "n"):
                    exit()  # exit the program if
            except ValueError:
                print("Error: Invalid option.\n")
                continue


if __name__ == '__main__':
    main()
