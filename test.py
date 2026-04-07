import os
import re
import subprocess
import ppadb.device
import tensorflow as tf
from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases

# --- THE ULTIMATE CONVERT-AND-PUSH MONKEYPATCH ---
def robust_convert_and_push(self, src, dest, mode=0o644, progress=None):
    # 1. Check if source is a SavedModel directory
    if os.path.isdir(src):
        tflite_path = src.rstrip('/') + ".tflite"
        print(f" [Monkeypatch] Converting SavedModel to TFLite: {src} -> {tflite_path}")
        
        # Perform the actual conversion
        converter = tf.lite.TFLiteConverter.from_saved_model(src)
        tflite_model = converter.convert()
        
        # Save the temporary .tflite file locally
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        src = tflite_path
        # Ensure the destination on Android also ends in .tflite
        if not dest.endswith('.tflite'):
            dest = dest.rstrip('/') + ".tflite"

    print(f" [Monkeypatch] PUSHING: {src} -> {dest}")
    subprocess.run(["adb", "-s", self.serial, "push", src, dest], check=True, capture_output=True)

def robust_adb_shell(self, command, handler=None, timeout=None):
    # Strip failing taskset mask
    if "taskset" in command:
        command = re.sub(r'taskset\s+\w+\s+', '', command)
    
    # Ensure the --graph path in the command points to the .tflite file
    graph_match = re.search(r'--graph=([^\s]+)', command)
    if graph_match:
        graph_path = graph_match.group(1)
        if not graph_path.endswith('.tflite'):
            command = command.replace(graph_path, graph_path.rstrip('/') + ".tflite")
        
    print(f" [Monkeypatch] EXECUTING: {command}")
    res = subprocess.run(["adb", "-s", self.serial, "shell", command], capture_output=True, text=True)
    return res.stdout

ppadb.device.Device.push = robust_convert_and_push
ppadb.device.Device.shell = robust_adb_shell
# ------------------------------------------------

# Initialize and Test
builder_config.init(workspace_path="./z839")
backend = connect_backend(backend_name='tflite_cpu')
testcases = generate_testcases()
single_test_model = testcases['BF_add_add']['add_1']['model'] 

try:
    result = backend.profile(single_test_model)
    print(f"\nSUCCESS! Latency: {result['latency'].avg} us")
except Exception as e:
    import traceback
    traceback.print_exc()