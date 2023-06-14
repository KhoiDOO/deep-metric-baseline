import torch
from torch import nn
from torchvision import models
import argparse
from .effb0 import effb0
from .resnet18 import rn18

model_map = {
    'effb0' : effb0,
    'resnet18' : rn18
}

def get_model(model:str, weight = True) -> nn.Module:
    if model not in list(model_map.keys()):
        raise Exception(f'the model {model} is current not supported')
    
    return model_map[model](
        weight = weight
    )