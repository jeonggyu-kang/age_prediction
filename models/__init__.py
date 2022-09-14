from .autoencoder import CXRAutoencoder2
from .age_predictor import AgePredictor
import torch

def get_model(model_name, ckpt_path = None, **kwargs):
    if model_name.lower() == "AgePredictor".lower():
        model = AgePredictor()
    elif model_name.lower() == "cxrautoencoder2":
        model = CXRAutoencoder2(**kwargs)
    else:
        print ('can not find model', model_name)
        exit(1)


    if ckpt_path is not None:
        print (f'Loading trained weight from {ckpt_path}..')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['weight'])

    model.cuda()

    return model