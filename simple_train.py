import os
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from pytorch_metric_learning import miners, losses, distances, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from dataset import get_dataset
from model import get_model
from utils import test_roc, blockPrint, enablePrint

def train(args: argparse):
    
    # Setup folder
    run_dir = os.getcwd() + "/runs"
    os.makedirs(run_dir, exist_ok = True)
    
    metric_dir = run_dir + f"/{args.metric}"
    os.makedirs(metric_dir, exist_ok = True)
    
    now = datetime.now().strftime("%m-%d-%Y - %H-%M-%S")
    save_dir = metric_dir + f"/{now}"
    os.makedirs(save_dir, exist_ok = True)
    
    args.log_path = save_dir + "/log.parquet"
    args.best_checkpoint = save_dir + "/best.pt"
    args.last_checkpoint = save_dir + "/last.pt"
    
    # Setup Multi GPU Training
    args.ngpus = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{args.port}'
    args.world_size = args.ngpus# Setup folder
    run_dir = os.getcwd() + "/runs"
    os.makedirs(run_dir, exist_ok = True)
    
    metric_dir = run_dir + f"/{args.metric}"
    os.makedirs(metric_dir, exist_ok = True)
    
    now = datetime.now().strftime("%m-%d-%Y - %H-%M-%S")
    save_dir = metric_dir + f"/{now}"
    os.makedirs(save_dir, exist_ok = True)
    
    args.log_dir = save_dir
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
    
    #Writer
    if args.rank == 0:
        writer = SummaryWriter(args.log_dir)
    
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
        vector_size=args.vs
    )
    model.cuda(gpu)
    model = DDP(model, device_ids=[gpu])
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    
    # Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs)
    
    # Loss 
    criterion = losses.TripletMarginLoss(
        margin=0.2,
        distance = distances.CosineSimilarity(),
        reducer=reducers.ThresholdReducer(low=0)
    )
    miner = miners.TripletMarginMiner(
        margin=0.2, 
        distance=distances.CosineSimilarity(), 
        type_of_triplets="semihard"
    )
    
    # Training
    old_loss = 1e26
    old_eer = 1
    for epoch in range(args.epochs):
        model.train()
        train_features = []
        train_labels = []
        for _, train_data in tqdm(enumerate(train_loader)):
            train_input, train_label = train_data
            train_input = train_input.cuda(gpu)
            train_label = train_label.cuda(gpu).long()
            train_feature = model(train_input)
            hard_pairs = miner(train_feature, train_label)
            train_loss = criterion(train_feature, train_label, hard_pairs)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            if args.rank == 0:
                train_features += train_feature.data.cpu().numpy().tolist()
                train_labels += train_label.data.cpu().numpy().tolist()
                
                blockPrint()
                writer.add_embedding(
                    mat=train_feature,
                    metadata=train_label.data.cpu().numpy().tolist(),
                    global_step=epoch,
                    tag='train'
                )
                enablePrint()
        
        if args.rank == 0:
            train_eer, train_ths, train_neg_score, train_pos_score = test_roc(train_features, train_labels)
            print(f"Epoch: {epoch}")
            print(f"Train - Loss: {train_loss.item()} - EER: {train_eer} - Threshold: {train_ths} - Negative Score: {train_neg_score} - Positive Score: {train_pos_score}")
            
            model.eval()
            with torch.no_grad():
                valid_features = []
                valid_labels = []
                for _, valid_data in tqdm(enumerate(test_loader)):
                    valid_input, valid_label = valid_data
                    valid_input = valid_input.cuda(gpu)
                    valid_label = valid_label.cuda(gpu).long()
                    valid_feature = model(valid_input)
                    valid_loss = criterion(valid_feature, valid_label)
                    
                    valid_features += valid_feature.data.cpu().numpy().tolist()
                    valid_labels += valid_label.data.cpu().numpy().tolist()
                    
                    blockPrint()
                    writer.add_embedding(
                        mat=train_feature,
                        metadata=train_label.data.cpu().numpy().tolist(),                        
                        global_step=epoch,
                        tag='valid'
                    )
                    enablePrint()
                
                valid_eer, valid_ths, valid_neg_score, valid_pos_score = test_roc(train_features, train_labels)
                
                print(f"Valid - Loss: {valid_loss.item()} - EER: {valid_eer} - Threshold: {valid_ths} - Negative Score: {valid_neg_score} - Positive Score: {valid_pos_score}")

            if valid_eer <= old_eer and valid_loss <= old_loss:
                best_save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    'eer': valid_eer
                }
                torch.save(best_save_dict, args.best_checkpoint)
            last_save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'eer': valid_eer
            }
            torch.save(last_save_dict, args.last_checkpoint)

    if args.rank == 0:
        writer.close()