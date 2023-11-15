from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights

from models.seg_res_net import SegResNet

import torch.nn as nn
import torch


def get_deeplabv3_mobilenet_v3_large(pretrained=False):
    if pretrained:
        model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)

        for i in range(13):
            for param in model.backbone[f"{i}"].parameters():
                param.requires_grad = False

    else:
        model = deeplabv3_mobilenet_v3_large()

    
    model.classifier[4] = nn.Conv2d(256, 68, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(10, 68, kernel_size=(1, 1), stride=(1, 1))

    return model


def get_segmentation_model(model_name, pretrained=False):
    if model_name == "DeepLabV3_MobileNet_V3_Large":
        return get_deeplabv3_mobilenet_v3_large(pretrained=pretrained)
    
    elif model_name == "SegResNet":
        return SegResNet(spatial_dims=2, in_channels=3, out_channels=68)
    else:
        raise Exception("Model not implemented")