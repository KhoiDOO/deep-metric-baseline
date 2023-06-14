import torch
from torch import nn
from torchvision import models

def effb6(num_input = None, num_class = 1000, weight = True, custom = False):
    model = models.efficientnet_b0(
        weights='IMAGENET1K_V1' if weight else 'DEFAULT'
    )
    if custom:
        if num_input is not None:
            model.features[0][0] = nn.Conv2d(num_input, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        del model.features[7:]
        del model.features[6][1:]
        del model.avgpool
        del model.classifier
        del model.features[6][0].stochastic_depth
        del model.features[6][0].block[3]    
        model.features.append(
            nn.Sequential(
                nn.Conv2d(672, 256, 3, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        model.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_class)
        )
    else:
        model.classifier[1] = nn.Linear(1280, num_class)
    return model