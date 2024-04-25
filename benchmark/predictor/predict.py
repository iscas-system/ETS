import torch
from torch.utils.data import DataLoader

from predictor.data import load_graph, load_scalers, Graph
from predictor.lstm import load_pretrained_predictor
from predictor.process import init_dataset, preprocess_dataset, to_device, compute_durations


def model_predict(env: str, measure_file: str)->list[Graph]:
    model = load_pretrained_predictor(env)
    model.eval()
    scalers = load_scalers(env)
    eval_graphs = [load_graph(measure_file)]
    eval_ds = init_dataset(eval_graphs)
    preprocessed_eval_ds = preprocess_dataset(eval_ds, scalers)
    dl = DataLoader(preprocessed_eval_ds, batch_size=128, shuffle=False)
    input_batchs, output_batchs = list(), list()
    for data in dl:
        features, labels = data
        features, labels = to_device("cuda", features, labels)
        with torch.no_grad():
            outputs = model(features)
        input_batchs.append(features)
        output_batchs.append(outputs)
    result = compute_durations(input_batchs, output_batchs, scalers, eval_graphs)
    return result
