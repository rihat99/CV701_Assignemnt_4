from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights

from torchvision.models import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

from torchvision.models import squeezenet1_1
from torchvision.models import SqueezeNet1_1_Weights

from torch import nn
import torch


def get_resnet50(pretrained=False):
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

    else:
        model = resnet50()
    

    # Add on fully connected layers for the output of our model

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=2048,
            out_features=68 * 2,
            bias=True
        )
    )
    
    return model


def get_resnet18(pretrained=False):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

    else:
        model = resnet18()

    # Add on fully connected layers for the output of our model

    model.fc = torch.nn.Sequential(
        nn.Dropout(0.2),
        torch.nn.Linear(
            in_features=512,
            out_features=68*2,
            bias=True
        )
    )

    return model


def get_mobilenet_v3_small(pretrained=False):
    if pretrained:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

    else:
        model = mobilenet_v3_small()

    # Add on fully connected layers for the output of our model

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=576,
            out_features=512,
            bias=True
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(
            in_features=512,
            out_features=68 * 2,
            bias=True
        )
    )

    return model


def get_mobilenet_v3_large(pretrained=False):
    if pretrained:
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

    else:
        model = mobilenet_v3_large()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=960,
            out_features=512,
            bias=True
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(
            in_features=512,
            out_features=68 * 2,
            bias=True
        )
    )

    return model

def get_efficientnet_v2_s(pretrained=False):
    if pretrained:
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(7):
            for param in model.features[i].parameters():
                param.requires_grad = False

    else:
        model = efficientnet_v2_s()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.2),
        torch.nn.Linear(
            in_features=1280,
            out_features=256,
            bias=True
        ),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(
            in_features=256,
            out_features=68 * 2,
            bias=True
        )
    )

    return model

def get_squeezenet1_1(pretrained=False):
    if pretrained:
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)

        # Freeze model weights
        for i in range(10):
            for param in model.features[i].parameters():
                param.requires_grad = False

    else:
        model = squeezenet1_1()

    # Add on fully connected layers for the output of our model
    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.5),
        torch.nn.Conv2d(
            in_channels=512,
            out_channels=68 * 2,
            kernel_size=1,
            stride=1
        ),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )

    return model


def get_model(model_name, pretrained=False):
    if model_name == "ResNet50":
        return get_resnet50(pretrained)
    elif model_name == "ResNet18":
        return get_resnet18(pretrained)
    elif model_name == "MobileNetV3Small":
        return get_mobilenet_v3_small(pretrained)
    elif model_name == "MobileNetV3Large":
        return get_mobilenet_v3_large(pretrained)
    elif model_name == "EfficientNetV2S":
        return get_efficientnet_v2_s(pretrained)
    elif model_name == "SqueezeNet1_1":
        return get_squeezenet1_1(pretrained)
    
    else:
        raise Exception("Model not implemented")