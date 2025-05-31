import torch.nn as nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights, ResNet18_Weights, VGG16_Weights


def resnet18():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def googlenet():
    weights = GoogLeNet_Weights.DEFAULT
    model = models.googlenet(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def vgg16():
    weights = VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final layer
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    for param in model.classifier[-1].parameters():
        param.requires_grad = True

    return model