import os
from pathlib import Path
from torchvision import datasets
from torchvision import transforms
import argparse
from .augmentation import CLSTransform, CLTransform

# Save data path
save_dir = "~/data/"

base_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Data Information
data_map = {
    "cifar10" : {
        '#class' : 10,
        'dataset' : datasets.CIFAR10,
        'img_size' : 32
    },
    'cifar100' : {
        '#class' : 100,
        'dataset' : datasets.CIFAR100,
        'img_size' : 32
    }
}

# Get Dataset
def get_dataset(args: argparse):
    if args.ds not in list(data_map.keys()):
        raise Exception(f"The data set {args.ds} is currently not supported")
    data_info = data_map[args.ds]
    class_cnt = data_info['#class']
    
    train_dataset = data_info['dataset'](
        root = save_dir,
        transform = CLSTransform(
            size=data_map[args.ds]['img_size']
        ),
        train = True,
        download = True
    )
    test_dataset = data_info['dataset'](
        root = save_dir,
        transform = base_test_transform,
        train = False,
        download = True
    )
    return (class_cnt, train_dataset, test_dataset)