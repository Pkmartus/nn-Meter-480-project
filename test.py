import monkeypatch # type: ignore
from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases


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