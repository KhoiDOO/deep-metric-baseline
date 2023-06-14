import os, sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Optimizer BenchMark',
                    description='This project takes into considering the performance comparison between optimizers',
                    epilog='ENJOY!!!')

    # Training settings
    parser.add_argument('--bs', type = int, default=32,
                    help='batch size')
    parser.add_argument('--workers', type = int, default=4,
                    help='Number of processor used in data loader')
    parser.add_argument('--epochs', type = int, default=1,
                    help='# Epochs used in training')
    parser.add_argument('--lr', type=float, default=0.01, 
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--port', type=int, default=8080, help='Multi-GPU Training Port.')
    parser.add_argument('--wd', default=5e-4, type=float, metavar='W',
                    help='weight decay')
    parser.add_argument('--ds', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Data set name')
    
    # Model settings
    parser.add_argument('--model', type=str, default='effb0', choices= ['effb0'],
                        help='model used in training')
    parser.add_argument('--weight', action='store_true', 
                        help='Toggled to use pretrained weight')
    parser.add_argument('--custom', action='store_true', 
                        help='Toggled to use custom model')
    
    # Metric settings
    parser.add_argument('--metric', type=str, default='arc_margin', choices=['add_margin', 'arc_margin', 'sphere'],
                        help="Type of metric used in training")
    parser.add_argument('--vs', type=int, default=128, 
                        help='Vector size use in embedding')
    parser.add_argument('--easy_margin', action='store_true', 
                        help='Toggled to use easy margin')
    parser.add_argument('--loss', type=str, default=None, choices=[None, 'focal_loss'],
                        help='loss used in training, set focal loss to use custom loss or leave to use \
                            cross entropy')
    parser.add_argument('--dv', nargs='+', default=-1,
                        help='List of devices used in training', required=True)
    
    args = parser.parse_args()
    
    if args.dv != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.dv)
    else:
        print("All GPU in use")
    
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)