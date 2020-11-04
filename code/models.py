"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Definition of Models

    * Implemented Models:
        - ResNet 18
        - EfficientNet B4
"""


from torch import nn
from torchvision import models
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


def fcn():
    """
        fcn FCN ResNet 50 model definition.
    """

    model = models.segmentation.fcn_resnet101(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # New output layers
    model.classifier = FCNHead(2048, 2)
    model.aux_classifier = FCNHead(2048, 2)

    return model


def deeplab():
    """
        deeplab DeepLab V3 - ResNet 101 model definition.
    """

    model = models.segmentation.deeplabv3_resnet101(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = DeepLabHead(2048, 2)

    return model
