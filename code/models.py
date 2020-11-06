"""
    Author: Mauro Mendez.
    Date: 02/11/2020.

    Definition of Models

    * Implemented Models:
        - UNet
        - FPN
        - DeeplLabV3+
"""


import segmentation_models_pytorch as smp


def unet(encoder, encoder_weights):
    model = smp.Unet(\
        encoder_name=encoder,\
        encoder_weights=encoder_weights,\
        classes=1,\
        activation='sigmoid')
    return model


def fpn(encoder, encoder_weights):
    model = smp.FPN(\
        encoder_name=encoder,\
        encoder_weights=encoder_weights,\
        classes=1,\
        activation='sigmoid')
    return model


def deeplab(encoder, encoder_weights):
    model = smp.DeepLabV3Plus(\
        encoder_name=encoder,\
        encoder_weights=encoder_weights,\
        classes=1,\
        activation='sigmoid')
    return model


def create_model(model, encoder, encoder_weights):
    if model == 'unet':
        return unet(encoder, encoder_weights)
    elif model == 'fpn':
        return fpn(encoder, encoder_weights)
    elif model == 'deeplab':
        return deeplab(encoder, encoder_weights)
