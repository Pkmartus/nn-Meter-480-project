import sys
import tf2onnx
from keras import models, Input, layers
import tensorflow as tf
from nn_meter import load_latency_predictor
import os

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
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.LeakyReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dense(128),
        layers.LeakyReLU(),
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

    @tf.function(input_signature=input_signature)
    def model_fn(x):
        return model(x, training=False)

    onnx_model, _ = tf2onnx.convert.from_function(
        model_fn,
        input_signature=input_signature,
        opset=13
    )

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


def clear():
    """
    Function that helps keep the CLI clean
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def pause():
    """
    Function that helps with pacing of CLI output
    """
    input(f"Press Enter to continue...")
    print()


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
            print(f"\nUsing model: {model_filename}\n")
            print('Options:')
            for i, platform in enumerate(hardware_inference_frameworks, start=1):
                print(f"{i}. {platform}")

            # Prompt user for input
            user_input = input(
                f"\nEnter a number (1-{num_frameworks}) to "
                "select a device inference framework to perform "
                "latency prediction on.\n\n"
                "Or, if you are trying to create your own NN-meter inference framework, please enter \"new\" \n\n"
                "Option: "
            )
            print("\n", end='')  # print newline
            # exit case if we wanna leave early
            if (user_input == '0'):
                sys.exit(0)

            try:
                # Check if user entered a non-numeric value or invalid option
                if ((not user_input.isnumeric) and user_input != "new"):
                    raise ValueError
                if (user_input == "new"):
                    break
                user_input = int(user_input)  # typecast input to integer
                if (user_input < 1 or user_input > num_frameworks):
                    raise ValueError
                else:
                    break  # valid input = exit the loop
            except:
                clear()
                print("Error: Invalid option.\n")
                continue

                # this can be placed within the main loop later just keeping it clean for now
        if (user_input == "new"):
            Loop = True
            while (Loop):

                print("Quick-Start Menu")
                print("1. Overiew")
                print("2. Environment Setup")
                print("3. Setting up your python venv")
                print("4. Installing other NN-meter dependinces")
                user_input = input(
                    f"\nEnter a number to select an option, or enter 0 to quit: ")

                if ((not user_input.isnumeric)):
                    print("Thats not a valid option")
                elif user_input == '0':
                    print(user_input)
                    Loop = False
                elif user_input == '1':  # Letting user know the general plan
                    clear()
                    print(r"""
.-------------------------------------------------------------------------------------------------------------------------------------.
|   _   _ _   _                          _                          _      _             _             _                 _     _      |
|  | \ | | \ | |          _ __ ___   ___| |_ ___ _ __    __ _ _   _(_) ___| | __     ___| |_ __ _ _ __| |_    __ _ _   _(_) __| | ___ |
|  |  \| |  \| |  _____  | '_ ` _ \ / _ | __/ _ | '__|  / _` | | | | |/ __| |/ _____/ __| __/ _` | '__| __|  / _` | | | | |/ _` |/ _ \|
|  | |\  | |\  | |_____| | | | | | |  __| ||  __| |    | (_| | |_| | | (__|   |_____\__ | || (_| | |  | |_  | (_| | |_| | | (_| |  __/|
|  |_| \_|_| \_|         |_| |_| |_|\___|\__\___|_|     \__, |\__,_|_|\___|_|\_\    |___/\__\__,_|_|   \__|  \__, |\__,_|_|\__,_|\___||
|                                                          |_|                                               |___/                    |
'-------------------------------------------------------------------------------------------------------------------------------------'
""")
                    print("="*75)
                    print("🛠️  NN-METER CUSTOM BUILDER: COMPREHENSIVE ROADMAP  🛠️")
                    print("="*75 + "\n")

                    # Step 1
                    print("STEP 1: ENVIRONMENT & TOOLCHAIN PREPARATION")
                    print("  - Objective: Get your development ecosystem together.")
                    print(
                        "  - Needs: Android Studio/SDK, Python virtual environment (v3.8+), ")
                    print("    and specific hardware drivers/compilers.")
                    print("  - Duration: ~1 hour\n")

                    # Step 2
                    print("STEP 2: HARDWARE ARCHITECTURE SPECIFICATION")
                    print("  - Objective: Formalize your device's capabilities.")
                    print("  - Needs: JSON descriptor files defining compute units, ")
                    print("    memory hierarchy, and clock speeds.")
                    print("  - Duration: ~1 hour\n")

                    # Step 3
                    print("STEP 3: IR PARSING & OP MAPPING")
                    print("  - Objective: Bridge your model to hardware instructions.")
                    print(
                        "  - Needs: Custom parser scripts to map framework IRs (e.g., ONNX/TFLite) ")
                    print("    to your target’s specific operational primitives.")
                    print("  - Duration: ~2-3 hours\n")

                    # Step 4
                    print("STEP 4: LATENCY PREDICTOR DEVELOPMENT")
                    print("  - Objective: Train your estimation engine.")
                    print(
                        "  - Needs: Performance data collection from your actual hardware ")
                    print("    to regress latency for target operators.")
                    print("  - Duration: ~2-4 hours\n")

                    # Step 5
                    print("STEP 5: INTEGRATION & VALIDATION")
                    print("  - Objective: Stitch and verify the workflow.")
                    print(
                        "  - Needs: Registering your builder in the NN-Meter framework and ")
                    print("    running smoke tests against a standard model suite.")
                    print("  - Duration: ~1-2 hours\n")

                    print("-" * 75)
                    print("⚡ TOTAL ESTIMATED EFFORT: 7 to 11 hours")
                    print("-" * 75 + "\n")

                    print("="*80)
                    print(" NN-METER: CUSTOM BUILDER ARCHITECTURE & FILE FORMATS ")
                    print("="*80 + "\n")

                    print("--- DIRECTORY STRUCTURE ---")
                    print(
                        "When you finalize your custom predictor, it should be organized as follows:")
                    print("  /customized_predictor/")
                    print(
                        "  ├── meta.yaml              # Configuration/metadata for registration")
                    print(
                        "  ├── fusion_rules.json      # Mapping of supported operator fusions")
                    print(
                        "  └── [kernel_name].pkl      # Latency predictor models (Pickle files) for each kernel\n")

                    print("--- KEY FILE FORMATS EXPLAINED ---")
                    print("1. meta.yaml")
                    print("   - Type: YAML")
                    print(
                        "   - Purpose: Acts as the manifest. It defines the 'name' for your predictor,")
                    print(
                        "     the 'category' (e.g., cpu, gpu, npu), and links to the predictor folder.")
                    print("\n2. fusion_rules.json")
                    print("   - Type: JSON")
                    print(
                        "   - Purpose: Contains the detected fusion rules that inform NN-Meter how to")
                    print(
                        "     group operators into kernels for your specific hardware architecture.")
                    print("\n3. [kernel_name].pkl")
                    print("   - Type: Python Pickle")
                    print(
                        "   - Purpose: Serialized machine learning models (typically trained regressors)")
                    print(
                        "     that output predicted latency given specific kernel input parameters.")
                    print("\n" + "="*80)
                    print(
                        "Pro-tip: Keep your [kernel_name].pkl files named exactly as they appear in")
                    print("your fusion rules to avoid runtime mapping errors")

                    pause()  # pause for user to read the overview
                    print()

                elif user_input == '2':  # Setup environment (python install, venv, and dependencies)
                    print("="*75)
                    print("NN-METER BUILDER ENVIRONMENT SETUP")
                    print("="*75 + "\n")
                    print("\n Open a new terminal and follow the steps below.")
                    pause()

                    # Step 1
                    print("STEP 1: CREATE BUILDER WORKSPACE")  
                    print("  - Command:")
                    print("    nn-meter create --tflite-workspace <path/to/place/workspace/>")
                    pause()

                    # Step 2
                    print("STEP 2: INSTALL PYTHON VERSION 3.9")
                    print("  - Source: https://www.python.org/downloads/")
                    pause()

                    # Step 3
                    print("STEP 3: CREATE VIRTUAL ENVIRONMENT") 
                    print("  - Linux Command:")
                    print("    python3.9 -m venv <env_name>")
                    print("  - Windows Command:")
                    print("    py 3.9 -m venv <env_name>")
                    pause()

                    # Step 4
                    print("STEP 4: ACTIVATE VIRTUAL ENVIRONMENT")
                    print("  - Linux Command:")
                    print("    source <env_name>/bin/activate")
                    print("  - Windows Command:")
                    print("    <env_name>\Scripts\activate")
                    pause()

                    # Step 5
                    print("STEP 5: INSTALL REQUIRED PACKAGES")
                    print("  - Commands:")
                    print("    pip install -r tool/docs/requirements/requirements.txt")
                    print("    pip install -r tool/docs/requirements/requirements_builder.txt")
                    pause()

                    # Step 6
                    print("STEP 6: INSTALL ANDROID STUDIO")
                    print("  - Source: https://developer.android.com/studio")
                    pause()

                    # Step 7
                    print("STEP 7: DOWNLOAD BENCHMARK MODEL VERSION 2.1")
                    print("  - Source: https://github.com/microsoft/nn-Meter/releases/tag/v2.0-data")
                    pause()

                elif user_input == '3':
                    # getting user to install right version of python. setting up
                    print("Getting Python and other tools up to date ")
                    # venv and such
                elif user_input == '4':
                    # getting the files downloaded for NN-meter
                    print("getting other dependecies set up ")
                elif user_input == '5':
                    # Walk the user through plugging in the device,
                    print("begining the process")
                    # enabling USB debugging, and to run "adb devices" to get the serial number.

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
