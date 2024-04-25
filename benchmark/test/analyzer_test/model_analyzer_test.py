from analyzer.model_analyzer import get_paramerters
from benchmark.models import load_resnet50

if __name__ == '__main__':
    resnet_50 = load_resnet50()
    # layers = get_layers(resnet_50)
    # for layer in layers:
    #     print(layer)
    print(get_paramerters(resnet_50))