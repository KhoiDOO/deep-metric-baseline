import os, sys
import argparse
from simple_train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Deep Metric Learning Baseline',
                    epilog='ENJOY!!!')

    # Training settings
    parser.add_argument('--bs', type = int, default=128,
                    help='batch size')
    parser.add_argument('--workers', type = int, default=2,
                    help='Number of processor used in data loader')
    parser.add_argument('--epochs', type = int, default=1,
                    help='# Epochs used in training')
    parser.add_argument('--lr', type=float, default=0.0001, 
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--port', type=int, default=5555, help='Multi-GPU Training Port.')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay')
    parser.add_argument('--ds', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Data set name')
    
    # Model settings
    parser.add_argument('--model', type=str, default='effb0', choices= ['effb0', 'resnet18'],
                        help='model used in training')
    parser.add_argument('--weight', action='store_true', 
                        help='Toggled to use pretrained weight')
    parser.add_argument('--vs', type=int, default=128,
                        help='embedding size')
    parser.add_argument('--dv', nargs='+',
                        help='List of devices used in training', required=True)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.dv)
    
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    train(args=args)