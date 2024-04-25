import torch


class InputType(object):
    types2id = {
        torch.FloatTensor: 1,
        torch.HalfTensor: 2,
        torch.DoubleTensor: 3,
    }
    id2types = {
        1: torch.FloatTensor,
        2: torch.HalfTensor,
        3: torch.DoubleTensor,
    }

    str2type = {
        'float': torch.FloatTensor,
        'half': torch.HalfTensor,
        'double': torch.DoubleTensor,
    }