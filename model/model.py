import torch
from torch import nn
from torchvision import models
import argparse
from .effb6 import effb6

model_map = {
    'effb6' : effb6
}

def get_model(model:str, num_input:int = None, num_classes:int = None, weight = False, custom = False) -> nn.Module:
    if model not in list(model_map.keys()):
        raise Exception(f'the model {model} is current not supported')
    
    if num_classes is not None:
        return model_map[model](
            num_classes = num_classes,
            weight = weight,
            custom = custom
        )
    else:
        return model_map[model](
            weight = weight,
            custom = custom
        )