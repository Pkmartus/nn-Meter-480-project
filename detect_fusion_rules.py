from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import profile_models
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases
from nn_meter.builder.backend_meta.fusion_rule_tester import detect_fusion_rule
import monkeypatch  # type: ignore # pylint: disable=unused-import


# initialize builder
builder_config.init(
    workspace_path="./z839_workspace"
)

# create testcases
origin_testcases = generate_testcases()

# connect to backend
backend = connect_backend(backend_name='tflite_cpu')

# run testcases and collect profiling results
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')

# determine fusion rools from results
detected_results = detect_fusion_rule(profiled_results)
