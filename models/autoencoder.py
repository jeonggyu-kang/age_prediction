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


class CXRAutoencoder2(nn.Module):
    def __init__(self, 
        z_common = 24, 
        z_age = 12,
        z_sex = 12,
        input_shape=(2, 3, 448, 448), 
    ):
        super(CXRAutoencoder2, self).__init__()

        _, _, org_height, org_width = input_shape
        bottle_neck_height = org_height // 32
        bottle_neck_width  = org_width  // 32

        print ('bottle_neck size: ', bottle_neck_height, bottle_neck_width)

        assert org_height >= 224 and org_width >= 224

        # encoder 
        self.encoder = modules.resnet18(pretrained = True)
        bottleneck_shape = _get_eleme_num(self.encoder, torch.randn(input_shape))

        self.z_sex = z_sex
        self.z_age = z_age
        self.z_common = z_common
        z_dim = z_sex + z_age + z_common

        
        # encoder fc    
        self.encoder_fc = nn.Linear(bottleneck_shape[0], z_dim)

        # decoder 
        self.decoder = modules.ResDeconv(
            block=modules.BasicBlock,
            global_avg_pool = True,
            z_all = z_dim,
            bottleneck_shape = (2048, bottle_neck_height, bottle_neck_width)
        )
        
        self.sex_classifier = nn.Sequential(
            nn.Linear(self.z_sex, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

        self.age_regresser = nn.Sequential(
            nn.Linear(self.z_age, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )       

    

    
    def forward(self, x): # x: x-ray
        latent_code = self.encoder(x)
        latent_code = latent_code.mean(-1).mean(-1) # (2048 X H x W) -> (2048)
        latent_code = self.encoder_fc(latent_code)

        x_hat = self.decoder(latent_code)

        y1_hat = self.age_regresser(latent_code[:, :self.z_age]) # latent_code N x 64
        y2_hat = self.sex_classifier(latent_code[:, self.z_age:self.z_sex + self.z_age]) # latent_code N x 64



        output_dict = {
            'x_hat' : x_hat, # reconstructed image
            'y1_hat' : y1_hat, # age
            'y2_hat' : y2_hat, # sex
            'latent_code' : latent_code
        }


        return output_dict

if __name__ == '__main__':
    
    model2 = CXRAutoencoder2(z_age = 24, z_sex = 24, z_common = 128, input_shape=(2,3,448*2, 448*2)).cuda()

    batch_size = 4
    image = torch.rand(batch_size, 3, 448*2, 448*2).cuda()


    output_dict2 = model2(image) # w/ bottleneck-linear



    for k, v in output_dict2.items():
        print(k, v.shape)
