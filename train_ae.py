# train process main script
import os
import cv2
import numpy as np

import torch

from models import get_model                     # "models" is a directory 

from config import get_hyperparameters            
from dataset import get_dataloader

from logger import get_logger
from runner_ae import trainer_ae
from visualizer import Umap


def main():
    # TODO : apply easydict
    args = get_hyperparameters()                 #             

    model = get_model(
        input_shape = (2, 3, *args['image_size']),
        model_name = args['model_name'],
        z_common = args['z_common'],
        z_age = args['z_age'],
        z_sex = args['z_sex']
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_mse = torch.nn.MSELoss()       

    if args.get('mile_stone') is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = args['mile_stone'], gamma = 0.1)            
    else:
        scheduler = None

    
    mode = 'train'

    train_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False
    )

    
    mode = 'test'
    test_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False
    )
    
    writer = get_logger(args['save_root'] )

    trainer_ae(                                      # from runner.py
        max_epoch = args['max_epoch'],
        model = model,
        train_loader = train_loader,
        test_loader = test_loader,

        loss_mse = loss_mse,
        loss_ce = loss_ce,
        optimizer = optimizer,
        scheduler = scheduler,
        meta = {
            'save_every' : 5,
            'print_every' : 5,
            'test_every' : 5
        },
        writer = writer,
        
    )


if __name__ == '__main__':
    main()

