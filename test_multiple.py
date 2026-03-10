import json
from nn_meter import load_latency_predictor

predictor = load_latency_predictor('myriadvpu_openvino2019r2', predictor_version=1.0)

with open('alexnet_googlenet_vgg.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line.strip())
        model_name = entry['model']
        graph      = entry['graph']          # already nn-Meter IR format

        latency_ms = predictor.predict(graph, model_type='nn-meter')
        print(f"{model_name}: {latency_ms:.2f} ms")