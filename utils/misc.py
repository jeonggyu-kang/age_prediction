import os, sys
import numpy as np 

def parse_args_from_config(config_path):

    import importlib.util
    
    spec = importlib.util.spec_from_file_location("get_hpyerparameters", config_path )

    modulevar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulevar)

    return modulevar


def tensor_rgb2bgr(image):
    permute = [2, 1, 0]
    return image[:, permute]


def print_dataset_dist(sample_dict, prefix):
    WARNING = '\033[93m'
    ENDC = '\033[0m'

    num_total_samples = 0

    for k, v in sample_dict.items():
        num_total_samples += v
    
    class_indices = list(sample_dict.keys())
    class_indices.sort()

    thres = 100 / len(sample_dict.keys())

    print('='*5, prefix, str(num_total_samples), '='*5)

    for class_idx in class_indices:
        k = class_idx
        v = sample_dict[class_idx]
        ratio = (v/num_total_samples) * 100
        message = '{} : {} ({:.4f}%)'.format(k, v, ratio)

        if ratio < thres:
            color = WARNING
        else:
            color = ENDC

        print ('{}{}{}'.format(color, message, ENDC))
    print ('')     

def tensor2numpy(tensor_image):
    return tensor_image.cpu().numpy().astype(np.float32).transpose(1,2,0)