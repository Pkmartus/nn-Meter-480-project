import os
import re
import subprocess
import ppadb.device
import tensorflow as tf

# --- THE ULTIMATE CONVERT-AND-PUSH MONKEYPATCH ---
def robust_convert_and_push(self, src, dest, mode=0o644, progress=None):
    # Check if source is a SavedModel directory
    if os.path.isdir(src):
        tflite_path = src.rstrip('/') + ".tflite"
        # Only convert if the .tflite doesn't exist to save time
        if not os.path.exists(tflite_path):
            converter = tf.lite.TFLiteConverter.from_saved_model(src)
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
        src = tflite_path
        if not dest.endswith('.tflite'):
            dest = dest.rstrip('/') + ".tflite"

    subprocess.run(["adb", "-s", self.serial, "push", src, dest], check=True, capture_output=True)

def robust_adb_shell(self, command, handler=None, timeout=None):
    # Strip failing taskset mask
    if "taskset" in command:
        command = re.sub(r'taskset\s+\w+\s+', '', command)
    # Redirect graph path to the .tflite file
    graph_match = re.search(r'--graph=([^\s]+)', command)
    if graph_match:
        graph_path = graph_match.group(1)
        if not graph_path.endswith('.tflite'):
            command = command.replace(graph_path, graph_path.rstrip('/') + ".tflite")
    
    res = subprocess.run(["adb", "-s", self.serial, "shell", command], capture_output=True, text=True)
    return res.stdout

ppadb.device.Device.push = robust_convert_and_push
ppadb.device.Device.shell = robust_adb_shell
# ------------------------------------------------