import logging

import torch


def get_output(model: torch.nn.Module):
    for layer in model.children():
        if hasattr(layer, 'out_features'):
            logging.info(layer)
            return layer.out_features


def get_tensor_dimensions_impl(model, layer, image_size, for_input=False):
    t_dims = None

    def _local_hook(_, _input, _output):
        nonlocal t_dims
        t_dims = _input[0].size() if for_input else _output.size()
        return _output

    layer.register_forward_hook(_local_hook)
    dummy_var = torch.zeros(1, 3, image_size, image_size)
    model(dummy_var)
    return t_dims
