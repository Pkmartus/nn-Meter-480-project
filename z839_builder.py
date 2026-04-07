from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend

#initialize builder
builder_config.init(
    workspace_path="./z839"
)

backend = connect_backend(backend_name='tflite_cpu')



