import logging
import random

import torch

from benchmark.models.input import get_in_channel, get_input
from utils.log import init_log
from benchmark.run import run_cuda

if __name__ == '__main__':
    init_log()
    for md in benchmark.models.load_trad_model.ModelDescriptions:
        if md.name != md.RESNET_50.name:
            continue
        batch = 4
        h = 64
        w = 64
        dt = torch.DoubleTensor
        if 'vit' in md.value.name:
            md.value.configs['image_size'] = 256
        out_features = random.randint(3, 1000)
        md.value.configs['num_classes'] = out_features
        model = benchmark.models.load_trad_model.load_trad_model(model=md.value.name, dtype=dt, configs=md.value.configs)

        in_channel = get_in_channel(model)
        input = get_input(type=md.value.type, batch=batch, in_channel=in_channel, h=h, w=w, dtype=dt)
        target = torch.randn(batch, out_features)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        logging.info(f'testing run model {md.value.name}, in_channel = {in_channel}, out_features={out_features}')
        run_cuda(model, 7, input, target, criterion, optimizer, md.value.name)
