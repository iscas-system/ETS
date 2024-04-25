import logging
from collections import namedtuple
from enum import Enum
import torch
import torchvision.models
import torchvision.models as models

IMAGE_CLASSFICATION_MODEL = 'image classification'
SEMANTIC_SEGMENTATION_MODEL = 'segmanic segmentation'
OBJECT_DETECTION_MODEL = 'object detection'
# type, if first layer is conv2d, we need to know the in_channel for input, and target_shape for output
ModelDescription = namedtuple(typename='ModelDescription', field_names=['name', 'type', 'configs'])


class ModelDescriptions(Enum):
    RESNET_18 = ModelDescription(name='resnet18', type=IMAGE_CLASSFICATION_MODEL,
                                 configs={'weights': None, 'num_classes': 1000})
    RESNET_50 = ModelDescription(name='resnet50', type=IMAGE_CLASSFICATION_MODEL,
                                 configs={'weights': None, 'num_classes': 1000})
    RESNET_34 = ModelDescription(name='resnet34', type=IMAGE_CLASSFICATION_MODEL,
                                 configs={'weights': None, 'num_classes': 1000})
    RESNET_101 = ModelDescription(name='resnet101', type=IMAGE_CLASSFICATION_MODEL,
                                  configs={'weights': None, 'num_classes': 1000})
    RESNET_152 = ModelDescription(name='resnet152', type=IMAGE_CLASSFICATION_MODEL,
                                  configs={'weights': None, 'num_classes': 1000})
    WIDE_RESNET_50_2 = ModelDescription(name='wide_resnet50_2', type=IMAGE_CLASSFICATION_MODEL,
                                        configs={'weights': None, 'num_classes': 1000})
    WIDE_RESNET_101_2 = ModelDescription(name='wide_resnet101_2', type=IMAGE_CLASSFICATION_MODEL,
                                         configs={'weights': None, 'num_classes': 1000})
    ALEXNET = ModelDescription(name='alexnet', type=IMAGE_CLASSFICATION_MODEL,
                               configs={'weights': None, 'num_classes': 1000})
    CONVNEXT_BASE = ModelDescription(name='convnext_base', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    CONVNEXT_TINY = ModelDescription(name='convnext_tiny', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    CONVNEXT_SMALL = ModelDescription(name='convnext_small', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    CONVNEXT_LARGE = ModelDescription(name='convnext_large', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    DENSENET_121 = ModelDescription(name='densenet121', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'num_classes': 1000})
    DENSENET_161 = ModelDescription(name='densenet161', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'num_classes': 1000})
    DENSENET_169 = ModelDescription(name='densenet169', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'num_classes': 1000})
    DENSENET_201 = ModelDescription(name='densenet201', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B0 = ModelDescription(name='efficientnet_b0', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B1 = ModelDescription(name='efficientnet_b1', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B2 = ModelDescription(name='efficientnet_b2', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B3 = ModelDescription(name='efficientnet_b3', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B4 = ModelDescription(name='efficientnet_b4', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B5 = ModelDescription(name='efficientnet_b5', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B6 = ModelDescription(name='efficientnet_b6', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_B7 = ModelDescription(name='efficientnet_b7', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_V2_L = ModelDescription(name='efficientnet_v2_l', type=IMAGE_CLASSFICATION_MODEL,
                                         configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_V2_M = ModelDescription(name='efficientnet_v2_m', type=IMAGE_CLASSFICATION_MODEL,
                                         configs={'weights': None, 'num_classes': 1000})
    EFFICIENTNET_V2_S = ModelDescription(name='efficientnet_v2_s', type=IMAGE_CLASSFICATION_MODEL,
                                         configs={'weights': None, 'num_classes': 1000})
    GOOLENET = ModelDescription(name='googlenet', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'aux_logits': False, 'num_classes': 1000})  # mark
    INCEPTION_V3 = ModelDescription(name='inception_v3', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'aux_logits': False, 'num_classes': 1000})
    MOBILENET_V2 = ModelDescription(name='mobilenet_v2', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'num_classes': 1000})
    MOBILENET_V3_LARGE = ModelDescription(name='mobilenet_v3_large', type=IMAGE_CLASSFICATION_MODEL,
                                          configs={'weights': None, 'num_classes': 1000})
    MOBILENET_V3_SMALL = ModelDescription(name='mobilenet_v3_small', type=IMAGE_CLASSFICATION_MODEL,
                                          configs={'weights': None, 'num_classes': 1000})
    MNASNET0_5 = ModelDescription(name='mnasnet0_5', type=IMAGE_CLASSFICATION_MODEL,
                                  configs={'weights': None, 'num_classes': 1000})
    MNASNET0_75 = ModelDescription(name='mnasnet0_75', type=IMAGE_CLASSFICATION_MODEL,
                                   configs={'weights': None, 'num_classes': 1000})
    MNASNET1_0 = ModelDescription(name='mnasnet1_0', type=IMAGE_CLASSFICATION_MODEL,
                                  configs={'weights': None, 'num_classes': 1000})
    MNASNET1_3 = ModelDescription(name='mnasnet1_3', type=IMAGE_CLASSFICATION_MODEL,
                                  configs={'weights': None, 'num_classes': 1000})
    REGNET_X_8GF = ModelDescription(name='regnet_x_8gf', type=IMAGE_CLASSFICATION_MODEL,
                                    configs={'weights': None, 'num_classes': 1000})
    REGNET_X_16GF = ModelDescription(name='regnet_x_16gf', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    REGNET_X_32GF = ModelDescription(name='regnet_x_32gf', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    REGNET_X_400MF = ModelDescription(name='regnet_x_400mf', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    REGNET_X_800MF = ModelDescription(name='regnet_x_800mf', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    REGNET_X_1_6_GF = ModelDescription(name='regnet_x_1_6gf', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    REGNET_X_3_2_GF = ModelDescription(name='regnet_x_3_2gf', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_16GF = ModelDescription(name='regnet_y_16gf', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_32GF = ModelDescription(name='regnet_y_32gf', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_128GF = ModelDescription(name='regnet_y_128gf', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_3_2_GF = ModelDescription(name='regnet_y_3_2gf', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_1_6_GF = ModelDescription(name='regnet_y_1_6gf', type=IMAGE_CLASSFICATION_MODEL,
                                       configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_400MF = ModelDescription(name='regnet_y_400mf', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    REGNET_Y_800MF = ModelDescription(name='regnet_y_800mf', type=IMAGE_CLASSFICATION_MODEL,
                                      configs={'weights': None, 'num_classes': 1000})
    SHUFFLENET_V2_X0_5 = ModelDescription(name='shufflenet_v2_x0_5', type=IMAGE_CLASSFICATION_MODEL,
                                          configs={'weights': None, 'num_classes': 1000})
    SHUFFLENET_V2_X1_0 = ModelDescription(name='shufflenet_v2_x1_0', type=IMAGE_CLASSFICATION_MODEL,
                                          configs={'weights': None, 'num_classes': 1000})
    SHUFFLENET_V2_X1_5 = ModelDescription(name='shufflenet_v2_x1_5', type=IMAGE_CLASSFICATION_MODEL,
                                          configs={'weights': None, 'num_classes': 1000})
    SHUFFLENET_V2_X2_0 = ModelDescription(name='shufflenet_v2_x2_0', type=IMAGE_CLASSFICATION_MODEL,
                                          configs={'weights': None, 'num_classes': 1000})
    SQUEEZENET1_0 = ModelDescription(name='squeezenet1_0', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    SQUEEZENET1_1 = ModelDescription(name='squeezenet1_1', type=IMAGE_CLASSFICATION_MODEL,
                                     configs={'weights': None, 'num_classes': 1000})
    SWIN_B = ModelDescription(name='swin_b', type=IMAGE_CLASSFICATION_MODEL,
                              configs={'weights': None, 'num_classes': 1000})
    SWIN_S = ModelDescription(name='swin_s', type=IMAGE_CLASSFICATION_MODEL,
                              configs={'weights': None, 'num_classes': 1000})
    SWIN_T = ModelDescription(name='swin_t', type=IMAGE_CLASSFICATION_MODEL,
                              configs={'weights': None, 'num_classes': 1000})
    SWIN_V2_B = ModelDescription(name='swin_v2_b', type=IMAGE_CLASSFICATION_MODEL,
                                 configs={'weights': None, 'num_classes': 1000})
    SWIN_V2_S = ModelDescription(name='swin_v2_s', type=IMAGE_CLASSFICATION_MODEL,
                                 configs={'weights': None, 'num_classes': 1000})
    SWIN_V2_T = ModelDescription(name='swin_v2_t', type=IMAGE_CLASSFICATION_MODEL,
                                 configs={'weights': None, 'num_classes': 1000})
    VGG11 = ModelDescription(name='vgg11', type=IMAGE_CLASSFICATION_MODEL,
                             configs={'weights': None, 'num_classes': 1000})
    VGG11_BN = ModelDescription(name='vgg11_bn', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VGG13 = ModelDescription(name='vgg13', type=IMAGE_CLASSFICATION_MODEL,
                             configs={'weights': None, 'num_classes': 1000})
    VGG13_BN = ModelDescription(name='vgg13_bn', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VGG16 = ModelDescription(name='vgg16', type=IMAGE_CLASSFICATION_MODEL,
                             configs={'weights': None, 'num_classes': 1000})
    VGG16_BN = ModelDescription(name='vgg16_bn', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VGG19 = ModelDescription(name='vgg19', type=IMAGE_CLASSFICATION_MODEL,
                             configs={'weights': None, 'num_classes': 1000})
    VGG19_BN = ModelDescription(name='vgg19_bn', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VIT_B_16 = ModelDescription(name='vit_b_16', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VIT_B_32 = ModelDescription(name='vit_b_32', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VIT_L_16 = ModelDescription(name='vit_l_16', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    VIT_L_32 = ModelDescription(name='vit_l_32', type=IMAGE_CLASSFICATION_MODEL,
                                configs={'weights': None, 'num_classes': 1000})
    # not working
    # VIT_H_14 = ModelDescription(name='vit_h_14', type=MODEL_CONV_TYPE, configs={'weights': None, 'num_classes': 1000})
    # MAXVIT_T = ModelDescription(name='maxvit_t', type=MODEL_CONV_TYPE, configs={'weights': None, 'num_classes': 1000})

    # semantic segmentation
    # return is a ordereddict
    # FCN_RESNET50 = ModelDescription(name='fcn_resnet50', type=SEMANTIC_SEGMENTATION_MODEL, configs={'weights': None, 'num_classes': 21})
    # FCN_RESNET101 = ModelDescription(name='fcn_resnet101', type=SEMANTIC_SEGMENTATION_MODEL, configs={'weights': None, 'num_classes': 21})
    DEEPLABV3_RESNET50 = ModelDescription(name='deeplabv3_resnet50', type=SEMANTIC_SEGMENTATION_MODEL,
                                          configs={'weights': None, 'num_classes': 21})
    DEEPLABV3_RESNET101 = ModelDescription(name='deeplabv3_resnet101', type=SEMANTIC_SEGMENTATION_MODEL,
                                           configs={'weights': None, 'num_classes': 21})


def load_model(name, configs):
    return torchvision.models.get_model(name, **configs)
    # models = __import__('torchvision', fromlist=['']).models
    # if name in models.__dict__:
    #     logging.info(f'loading model {name}')
    #     model = models.__dict__[name](weights=None, model_params)
    #     return model
    # else:
    #     print(f'no model {name}')


def load_trad_model(**params):
    model = load_model(params['model'], params['configs'])
    if params['dtype'] == torch.FloatTensor:
        return model.float()
    elif params['dtype'] == torch.HalfTensor:
        return model.half()
    elif params['dtype'] == torch.DoubleTensor:
        return model.double()
    return model


def load_nlp_model(**params):
    def get_nlp_layer(**params):
        if params['model'] == 'rnn':
            return torch.nn.RNN(input_size=params['input_size'], hidden_size=params['hidden_size'],
                                num_layers=params['num_layers'],
                                bidirectional=params['bidirectional'])
        elif params['model'] == 'lstm':
            return torch.nn.LSTM(input_size=params['input_size'], hidden_size=params['hidden_size'],
                                 num_layers=params['num_layers'],
                                 bidirectional=params['bidirectional'])
        elif params['model'] == 'gru':
            return torch.nn.GRU(input_size=params['input_size'], hidden_size=params['hidden_size'],
                                num_layers=params['num_layers'],
                                bidirectional=params['bidirectional'])

    class NLPModel(torch.nn.Module):
        def __init__(self, **params):
            super(NLPModel, self).__init__()
            self.model = get_nlp_layer(**params)
            num_directions = 2 if params['bidirectional'] else 1
            self.linear = torch.nn.Linear(in_features=params['hidden_size'] * num_directions,
                                          out_features=params['num_classes'])

        def forward(self, x):
            x, y = self.model(x)
            x = self.linear(x)
            return x

    return NLPModel(**params)


# Test Load
if __name__ == '__main__':
    for md in ModelDescriptions:
        model = load_model(md.value.name)
