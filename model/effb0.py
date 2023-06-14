import torch
from torch import nn
from torchvision import models

def effb0(weight = True):
    model = models.efficientnet_b0(
        weights='IMAGENET1K_V1' if weight else 'DEFAULT'
    )
    num_feature = model.classifier[1].in_features
    del model.classifier[1]
    
    return (num_feature, model)