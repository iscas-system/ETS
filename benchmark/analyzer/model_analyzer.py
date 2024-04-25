import torch


def get_layers(model: torch.nn.Module):
    """
    get layers
    :param model:
    :return:
    """
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_layers(child))
            except TypeError:
                flatt_children.append(get_layers(child))
    return flatt_children

def get_paramerters(layer: torch.nn.Module):
    return dict(layer.named_modules())

def get_input_shape(model: torch.nn.Module):
    # for Conv2D
    N, C = list(model.parameters())[0].size()
    return (N, C)
