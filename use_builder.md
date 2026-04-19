1. create builder workspace by running: nn-meter create --tflite-workspace <path/to/place/workspace/>
2. download python3.9
3. create virtual environment: python3.9 -m venv .z839_builder                        
4. activate venv source .z839_builder/bin/activate
5. install requirements:
    pip install -r tool/docs/requirements/requirements_builder.txt
    pip install -r tool/docs/requirements/requirements.txt
    (not sure which of these is needed)
6. install android studio (yes really wow)
    install sdk
7. download benchmark model (2.1 version) from https://github.com/microsoft/nn-Meter/releases/tag/v2.0-data
8. find device: adb devices
9. push to device:
    adb [-s <device-serial>] push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp

    # add executable permission to benchmark_model
    adb shell chmod +x /data/local/tmp/benchmark_model
10. edit backend_config.yaml with serial and correct paths
11. Test connection:
    nn-meter connect --backend <backend-name> --workspace <path/to/workspace>
12.run detect_fusion_rules.py
    currently wrestling with all latency outputting as 0
    gemini helped write helper functions to circomvent issues with device compatability
        device had less cores than nn-meter expected.
        tests were being generated as folders rather than .tflite files, a converter was needed
        ppadb was having issues copying data so a generated helper function replaces some of those operations.
13. moved the monkeypatch to it's own file
14. tried running build_latency predictor failed because od a keras version error
15. downgraded packages and reran
16. successfully created predictor for conv-bn-relu
17. created custom script to loop through kernel types in case of crash

