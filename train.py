# train process main script
import os
import cv2
import numpy as np

<<<<<<< HEAD

=======
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
>>>>>>> 7e70fcd8193f0a3a6226ce3c49387db63de0e1cb

import torch

from models import get_model                     # "models" is a directory 

from config import get_hyperparameters            
from dataset import get_dataloader

from logger import get_logger
from runner import trainer
from visualizer import Umap


def main():
    # TODO : apply easydict
    args = get_hyperparameters()                 #             

    model = get_model(
        global_avg_pool = args['global_avg_pool'],
        z_dim = args['z_dim'],
        z_cac = args['z_cac'],
        input_size = (2,3, *args['image_size']),
        n_class = args['n_class']
    )    

    if args['train_target'] == 'classifier':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args['learning_rate'])
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False

        try:
            for param in model.encoder_fc.parameters():
                param.requires_grad = False
            
        except:
            pass

        loss_recon = None
        loss_ce = torch.nn.CrossEntropyLoss()

    elif args['train_target'] == 'joint':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
        loss_recon = torch.nn.MSELoss()
        loss_ce = torch.nn.CrossEntropyLoss()

    elif args['train_target'] == 'fine-tune':
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters())
            +list(model.encoder_fc.parameters())
            +list(model.classifier.parameters()),            
             lr=args['learning_rate'])

        for param in model.decoder.parameters():
                param.requires_grad = False

        loss_recon = None
        loss_ce = torch.nn.CrossEntropyLoss()


    if args.get('mile_stone') is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones = args['mile_stone'], gamma = 0.1)            
    else:
        scheduler = None

    
    mode = 'train'
    '''
    small_loader = get_dataloader(
        dataset = args['dataset'],
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = args['batch_size'],
        num_workers = args['workers_per_gpu'],
        pipeline = args[mode]['pipeline'],
        csv = False,
        small_set = True
    )

    '''
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

    visualizer = Umap()

    trainer(                                      # from runner.py
        max_epoch = args['max_epoch'],
        model = model,
        train_loader = train_loader,
        test_loader = test_loader,
        # train_loader = small_loader,
        # test_loader = small_loader,
        loss_recon = loss_recon,
        loss_ce = loss_ce,
        optimizer = optimizer,
        scheduler = scheduler,
        meta = {
            'save_every' : 5,
            'print_every' : 5,
            'test_every' : 5
        },
        writer = writer,
        visualizer = visualizer
        
    )


if __name__ == '__main__':
    main()

