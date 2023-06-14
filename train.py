import os
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import time
import numpy as np

import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import get_dataset
from model import get_model
from metrics import *
from loss import FocalLoss
from utils import test_roc

def main(args: argparse):
    
    # Setup folder
    run_dir = os.getcwd() + "/runs"
    os.makedirs(run_dir, exist_ok = True)
    
    metric_dir = run_dir + f"/{args.metric}"
    os.makedirs(metric_dir, exist_ok = True)
    
    now = now.strftime("%m-%d-%Y - %H-%M-%S")
    save_dir = metric_dir + f"/{now}"
    os.makedirs(save_dir, exist_ok = True)
    
    args.log_path = save_dir + "/log.parquet"
    args.best_checkpoint = save_dir + "/best.pt"
    args.last_checkpoint = save_dir + "/last.pt"
    
    # Setup Multi GPU Training
    args.ngpus = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{args.port}'
    args.world_size = args.ngpus
    
    # Save config
    config_path = save_dir + "/config.json"
    with open(config_path, mode='w') as file:
        json.dump(vars(args), file)
    
    mp.spawn(main_worker, (args,), args.ngpus)

def main_worker(gpu, args):
    args.rank += gpu
    
    dist.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank
    )

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    # Data Loader
    num_classes, train_dataset, test_dataset = get_dataset(args=args)
    
    assert args.bs % args.world_size == 0
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    per_device_batch_size = args.bs // args.world_size

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=per_device_batch_size, num_workers=args.workers, pin_memory=True, sampler=test_sampler
    )
    
    # Model
    model = get_model(
        model=args.model,
        weight=args.weight,
        custom=False
    ).cuda(gpu)
    num_feature = model.classifier[1].in_features
    del model.classifier[1]
    
    # Metric
    if args.metric == 'add_margin':
        metric_fc = AddMarginProduct(num_feature, num_classes, s=30, m=0.35)
    elif args.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(num_feature, num_classes, s=30, m=0.5, easy_margin=args.easy_margin)
    elif args.metric == 'sphere':
        metric_fc = SphereProduct(num_feature, num_classes, m=4)
    else:
        # metric_fc = nn.Linear(num_feature, num_classes)
        raise ValueError(f"metric must be one of ['add_margin', 'arc_margin', 'sphere'], found {args.metric} instead")
    
    metric_fc.cuda(gpu)
    metric_fc = DDP(metric_fc)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters()}, 
            {'params': metric_fc.parameters()}
        ],
        lr=args.lr,
        weight_decay=args.wd
    )
    
    #Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch)
    
    # Loss
    if args.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Training
    
    old_loss = 1e26
    old_acc = 0
    for epoch in range(args.epochs):
        model.train()
        for _, train_data in tqdm(enumerate(train_loader)):
            train_input, train_label = train_data
            train_input = train_input.cuda(gpu)
            train_label = train_label.cuda(gpu).long()
            train_feature = model(train_input)
            train_output = metric_fc(train_feature, train_label)
            train_loss = criterion(train_output, train_label)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            scheduler.step()
        
        if args.rank == 0:
            train_output = train_output.data.cpu().numpy()
            train_output = np.argmax(train_output, axis=1)
            train_label = train_label.data.cpu().numpy()
            train_acc = np.mean((train_output == train_label).astype(int))
            print(f"Epoch: {epoch} - Train Loss: {train_loss.item()} - Train Accuracy: {train_acc.item()}")
            
            model.eval()
            with torch.no_grad():
                for _, valid_data in tqdm(enumerate(test_loader)):
                    valid_input, valid_label = valid_data
                    valid_input = valid_input.cuda(gpu)
                    valid_label = valid_label.cuda(gpu).long()
                    valid_feature = model(valid_input)
                    valid_output = metric_fc(valid_feature, valid_label)
                    valid_loss = criterion(valid_output, valid_label).item()
            
            valid_output = valid_output.data.cpu().numpy()
            valid_output = np.argmax(valid_output, axis=1)
            valid_label = valid_label.data.cpu().numpy()
            valid_acc = np.mean((valid_output == valid_label).astype(int)).item()
                    
            print(f"Epoch: {epoch} - Valid Loss: {valid_loss} - Valid Accuracy: {valid_acc}")

            if valid_acc >= old_acc and valid_loss <= old_loss:
                best_save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    'val_acc': valid_acc
                }
                torch.save(best_save_dict, args.best_checkpoint)
            last_save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'val_acc': valid_acc
            }
            torch.save(best_save_dict, args.last_checkpoint)
    
    # Evaluation
    
    # Get best model
    model.load_state_dict(best_save_dict['model_state_dict'])
    
    model.eval()
    test_features = []
    test_labels = []
    with torch.no_grad():
        for _, valid_data in tqdm(enumerate(test_loader)):
            valid_input, valid_label = valid_data
            valid_input = valid_input.cuda(gpu)
            valid_label = valid_label.cuda(gpu).long()
            valid_feature = model(valid_input)
            
            test_features.append(valid_feature.data.cpu().numpy())
            test_labels.append(valid_label.data.cpu().numpy())
            
    eer, ths, neg_score, pos_score = test_roc(test_features, test_labels)
    
    print("Evaluation Result")
    print(f"EER: {eer} - Threshold: {ths} - Negative Score: {neg_score} - Positive Score: {pos_score}")