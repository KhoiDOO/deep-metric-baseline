import torch
from torch import nn
from torchvision import models

def effb0(weight = True, vector_size = 128):
    model = models.efficientnet_b0(
        weights='IMAGENET1K_V1' if weight else 'DEFAULT'
    )
    out_features = model.classifier[1].in_features
    # del model.classifier
    model.classifier == nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(out_features, vector_size, bias=True),
        nn.BatchNorm1d(vector_size),
        torch.nn.PReLU()
    )
    
    return model