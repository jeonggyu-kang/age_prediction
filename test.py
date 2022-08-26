import os 
import copy 
import argparse 

from utils import parse_args_from_config 
from models import get_model 
from logger import get_logger 
from runner import tester 
from dataset import get_dataloader
from visualizer import Umap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', type=str, default=None, help='Path to log directory')
    _args = parser.parse_args()

    if not os.path.exists(_args.log_root):
        print('Can not find directory {}'.format(_args.log_root))
        exit()


    _args.config = os.path.join(_args.log_root, 'config.py')
    modulevar = parse_args_from_config(_args.config)

    args = modulevar.get_hyperparameters(config=_args.config)


    

    ckpt_path = os.path.join(_args.log_root, 'best.pt')
    model = get_model(
        global_avg_pool = args['global_avg_pool'],
        z_dim = args['z_dim'],
        z_cac = args['z_cac'],
        input_size = (2, 3, *args['image_size']),
        n_class = args['n_class'],
        ckpt_path = ckpt_path
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

    save_path = os.path.join(__args.log_root, 'eval')
    writer = get_logger(save_path)

    visualizer = Umap()


    tester(
        model = model,
        test_loader = test_loader,
        writer = writer,
        visualizer = visualizer,
        confusion_matrix = True
    )



if __name__ == '__main__':
    main()