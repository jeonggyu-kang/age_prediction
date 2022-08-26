import torch
import torch.nn as nn
import numpy as np 


try:
    from . import modules
except:
    import modules


def _get_eleme_num(model, x):
    y = model(x)
    _, C, H, W = y.shape
    return (C,H,W)


class AgePredictor(nn.Module):
    def __init__(self, 
        input_shape=(2, 3, 448, 448), 
    ):
        super(AgePredictor, self).__init__()

        # feature extractor
        self.feature_extractor = modules.resnet18(pretrained = True)
        
    
        self.sex_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        self.age_regresser = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    

    
    def forward(self, x): # x: x-ray
        feature = self.feature_extractor(x)
        feature = feature.mean(-1).mean(-1) # (2048 X H x W) -> (2048)

        age_hat = self.age_regresser(feature)
        sex_hat = self.sex_classifier(feature)

        output_dict = {
            'age_hat' : age_hat,
            'sex_hat' : sex_hat
        }
        

        return output_dict

if __name__ == '__main__':
    model = AgePredictor()

    dummy_input = torch.rand(2,3,896,896)

    out_dict = model(dummy_input)

    print(out_dict['age_hat'], out_dict['sex_hat'])