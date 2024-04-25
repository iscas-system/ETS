import torch


def get_in_channel(model: torch.nn.Module):
    for child in model.modules():
        if type(child).__name__ == 'Conv2d':
            return child.weight.size(1)
    return -1


def get_input(**params):
    return get_conv_input(**params)


def get_conv_input(**params):
    # for resnet inchannel is 3
    data = torch.randn((params['batch'], params['in_channel'], params['h'], params['w']))
    if params['dtype'] == torch.FloatTensor:
        return data.float()
    # elif params['dtype'] == torch.LongTensor:
    #     return data.long()
    # elif params['dtype'] == torch.IntTensor:
    #     return data.int()
    # elif params['dtype'] == torch.ShortTensor:
    #     return data.short()
    elif params['dtype'] == torch.HalfTensor:
        return data.half()
    elif params['dtype'] == torch.DoubleTensor:
        return data.double()
    return data
