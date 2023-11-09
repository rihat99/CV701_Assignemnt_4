from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torch import nn
import torch


def get_resnet50(pretrained=False):
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

    else:
        model = resnet50()
    

    # Add on fully connected layers for the output of our model

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
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


def get_resnet18(pretrained=False):
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V2)

        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False

    else:
        model = resnet18()

    # Add on fully connected layers for the output of our model

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=512,
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


def get_model(model_name, pretrained=False):
    if model_name == "ResNet50":
        return get_resnet50(pretrained)
    elif model_name == "ResNet18":
        return get_resnet18(pretrained)
    else:
        raise Exception("Model not implemented")