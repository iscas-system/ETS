import torch

import pyprof

from benchmark.run import _run
from benchmark.models.load import load_model

pyprof.init()



def run_model(model: torch.nn.Module, batch_size: int, n: int, model_info: str):
    """
    run model n times
    :param model:
    :param batch_size:
    :param n:
    :return:
    """

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # create input and label
    x = torch.rand(batch_size, 3, 224, 224).cuda()
    target = torch.empty(batch_size, dtype=torch.long).random_(1000).cuda()

    _run(model, n, x, target, criterion, optimizer,model_info)


if __name__ == '__main__':
    resnet50 = load_model('resnet50', configs={'weights': None})
    batch = 16
    hw = [224, 224]
    n = 5
    dtype = torch.FloatTensor
    # todo alter_seq的合并为一个
    model_info = f'model=resnet50, batch={batch}, h={hw[0]},w={hw[1]}, dtype={dtype},'
    # train
    run_model(resnet50, batch, n, model_info)