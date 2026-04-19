#initialize builder config with workspace
import monkeypatch # type: ignore
from nn_meter.builder import builder_config
workspace = "./z839"
builder_config.init(workspace)

# build latency predictor for kernel
from nn_meter.builder import build_predictor_for_kernel
kernel_type = "conv-bn-relu"
backend = "tflite_cpu"

predictor, data = build_predictor_for_kernel(
    kernel_type, backend, init_sample_num=5, finegrained_sample_num=10, iteration=5, error_threshold = 0.1
)