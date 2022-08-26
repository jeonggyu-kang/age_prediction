from .autoencoder import CXRAutoencoder
import torch

def get_model(ckpt_path = None):

    model = AgePredictor()


    if ckpt_path is not None:
        print (f'Loading trained weight from {ckpt_path}..')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['weight'])

    model.cuda()

    return model