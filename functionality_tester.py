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

# Define hardware and inference frameworks supported by nn-Meter
hardware_inference_frameworks = [
    'cortexA76cpu_tflite21',
    'adreno640gpu_tflite21',
    'adreno630gpu_tflite21',
    'myriadvpu_openvino2019r2'
]
num_frameworks = len(hardware_inference_frameworks)


def createModel():
    """"
    Defines a CNN model architecture to be used by nn-Meter for latency 
    prediction.

    Returns: keras.src.engine.sequential.Sequential: the TensorFlow 
    Keras CNN model.
    """
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
    """
    Converts a TensorFlow Keras convolutional neural network model into 
    an ONNX format for nn-Meter latency prediction given a specified 
    device inference framework.

    Args: 
        model (keras.src.engine.sequential.Sequential): The TensorFlow 
         Keras model object to be used for latency prediction.
        filename (str): The filename of the model being converted to
         ONNX format.

    Returns: The filename of the model being converted to ONNX format.
    """
    # Add .onnx file extension to the model filename
    filename += ".onnx"

    # Use a batch size of 1 to simulate single-image edge inference
    input_signature = [tf.TensorSpec([1, 32, 32, 3], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature, opset=13)

    # Save the ONNX file (write binary)
    with open(filename, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    return filename


def mapModelToFileExt():
    """
    TODO: Write mapModelToFileExt function that maps model file 
    extension to variable for model_type parameter used by 
    nnMeterPredictor object (predictor.predict).
    """
    pass


def main():
    """
    Handles all user input and ouput. Allows the user to predict the
    latency of a convolutional neural network model on 1 of 4 different
    device inference frameworks.
    """
    print("\n", end='')  # print newline
    # Define CNN model architecture
    model = createModel()

    """
    TODO: Change functionality to allow users to enter their own custom
    model filename for use instead of current hardcoded method.
    """
    model_name = 'custom_model'

    """
    TODO: Implement mapModelToFileExt function to get the model file 
    extension. This will be used when setting the model_type parameter 
    used by the nnMeterPredictor object (predictor.predict).
    """

    # Convert model to onnx format
    model_filename = convertToONNX(model, model_name)

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
            print("\n", end='')  # print newline

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
                "Would you like to predict the latency of another device "
                "inference framework? Enter y/n: "
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
                    exit()  # exit the program
            except ValueError:
                print("Error: Invalid option.\n")
                continue


# Declare entrypoint of program
if __name__ == '__main__':
    main()
