import torch
from torch import nn
from torchvision import models

def rn18(weight = True):
    model = models.resnet18(
        weights='IMAGENET1K_V1' if weight else 'DEFAULT'
    )
    num_feature = model.fc.in_features
    del model.fc
    
    return (num_feature, model)