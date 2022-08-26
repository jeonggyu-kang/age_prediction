import os 
import copy 
import argparse 


from utils import parse_args_from_config 
from models import get_model 
from logger import get_logger 
from runner import grad_cam
from dataset import get_dataloader

from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image

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


    for model_name in args['name']:              # loop in various models
        log_root = os.path.join(_args.log_root, model_name)

        ckpt_path = os.path.join(log_root, 'best.pt')
        input_size = args['image_size']
        model = get_model(model_name, args['n_class'], ckpt_path = ckpt_path, input_size = (2,3,*input_size))
        model.eval()

        if args['loss'] in ['ce', 'cross-entropy']:
            task_type = 'classification'

        elif args['loss'] in ['mse', 'mean_squared_error']:
            task_type = 'regression'

        else:
            raise ValueError


        mode = 'test'
        test_loader = get_dataloader(
            dataset = args['dataset'],
            data_dir = args[mode]['img_dir'],
            ann_path = args[mode]['ann_file'],
            mode = mode,
            batch_size = 1,
            num_workers = 0,
            pipeline = args[mode]['pipeline'],
            csv = True
        )

        save_path = os.path.join(log_root, 'eval')
        writer = get_logger(save_path, use_cam=True)


        print(model)
        # exit()

        # target layers
        if 'vgg' in model_name:
            target_layers = [model.feature_extractor.features[-3]]

        elif 'hrnet' in model_name:
            target_layers =[model.feature_extractor.downsamp_modules[-1][-3]]    
        else:
            print('Please speicfy target layer for your model.')
            exit()
            
        cam = GradCAM(
            model = model,
            target_layers = target_layers,
            use_cuda = True
        )

        grad_cam(                        
            model = model,    
            data_loader = test_loader,
            writer = writer,
            cam = cam,
            export_csv = True,
            n_class = args['n_class'],
            task_type = task_type
        )


if __name__ == '__main__':
    main()