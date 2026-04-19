import os
import monkeypatch # type: ignore
from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.nn_meter_builder import build_predictor_for_kernel

workspace = "./z839_workspace"
builder_config.init(workspace)

# Instantiate the backend connection once
print("\n[QUEUE] Connecting to ADB backend...")
backend = "tflite_cpu"

# The full suite of standard nn-Meter operators
all_kernels = [
    "conv-bn-relu", 
    "dwconv-bn-relu", 
    "maxpool", 
    "avgpool", 
    "fc", 
    "concat", 
    "add", 
    "split", 
    "channelshuffle", 
    "se", 
    "global-avgpool",
    "hswish",
    "hsigmoid"
]

# Set up the directory checker
predictors_dir = os.path.join(workspace, "predictor_build", "results", "predictors")
os.makedirs(predictors_dir, exist_ok=True)

print(f"\n[QUEUE] Scanning {len(all_kernels)} operators...")

for kernel in all_kernels:
    # 1. AUTO-SKIP LOGIC: Check if we already have a compiled .pkl file
    existing_files = os.listdir(predictors_dir)
    if any(f.startswith(kernel) and f.endswith(".pkl") for f in existing_files):
        print(f"\n[QUEUE] Skipping '{kernel}' -> Already compiled!")
        continue
        
    print(f"\n=======================================================")
    print(f"[QUEUE] Starting active learning loop for: {kernel}")
    print(f"=======================================================")
    
    # 2. CRASH-PROOF BUILDER LOGIC
    try:
        build_predictor_for_kernel(
            kernel_type=kernel,
            backend=backend,
            init_sample_num=1000,      # Default prior samples
            finegrained_sample_num=10, # Default finegrained samples
            iteration=5,               # Max iterations
            error_threshold=0.1        # 10% error threshold target
        )
        print(f"\n[QUEUE SUCCESS] Successfully built predictor for {kernel}!")
        
    except Exception as e:
        print(f"\n[QUEUE ERROR] Kernel '{kernel}' crashed: {e}")
        print(f"[QUEUE ERROR] Moving to the next operator to prevent pipeline freeze...\n")

print("\n[QUEUE] ALL OPERATORS PROCESSED!")