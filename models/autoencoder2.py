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


class CXRAutoencoder(nn.Module):
    def __init__(self, 
        global_avg_pool, 
        z_dim = 512, 
        input_shape=(2, 3, 448, 448), 
        n_class=None,
        z_cac = None
    ):
        super(CXRAutoencoder, self).__init__()
        self.z_cac = z_cac

        self.encoder = modules.resnet50(pretrained = True)
        bottleneck_shape = _get_eleme_num(self.encoder, torch.randn(input_shape))

        self.global_avg_pool = global_avg_pool
        if self.global_avg_pool:
            self.encoder_fc = nn.Linear(bottleneck_shape[0], z_dim)
            self.decoder = modules.ResDeconv(
                block=modules.BasicBlock,
                global_avg_pool = True,
                z_all = z_dim,
                bottleneck_shape = bottleneck_shape
            )
        else:
            self.decoder = modules.ResDeconv(modules.BasicBlock)
        
        assert n_class is not None
        if self.global_avg_pool:

            if z_cac is None:

                self.classifier = nn.Sequential(
                    nn.Linear(z_dim, 256),
                    nn.ReLU(True),
                    nn.Linear(256, n_class)
                )

            else:
                assert isinstance(z_cac, int)
                self.classifier = nn.Sequential(
                    nn.Linear(z_cac, 64),
                    nn.ReLU(True),
                    nn.Linear(64, n_class)
                )
    
        else:

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(bottleneck_shape[0] * bottleneck_shape[1] * bottleneck_shape[2], 256),
                nn.ReLU(True),
                nn.Linear(256, n_class)
            )



    
    def forward(self, x): # x: x-ray
        latent_code = self.encoder(x)
        if self.global_avg_pool:
            latent_code = latent_code.mean(-1).mean(-1) # (2048 X 14 x 14) -> (2048)
            latent_code = self.encoder_fc(latent_code)

        x_hat = self.decoder(latent_code)

        if self.z_cac is None:
            y_hat = self.classifier(latent_code)

        else:
            y_hat = self.classifier(latent_code[:, :self.z_cac])

        output_dict = {
            'x_hat' : x_hat,
            'y_hat' : y_hat,
            'latent_code' : latent_code
        }

        return output_dict

if __name__ == '__main__':
    #model1 = CXRAutoencoder(n_class=5, use_classifier=True, global_avg_pool = False, input_shape=(2,3,448*2, 448*2)).cuda()
    model2 = CXRAutoencoder(n_class=5, use_classifier=True, global_avg_pool = True, z_dim = 256, input_shape=(2,3,448*2, 448*2)).cuda()

    batch_size = 4
    image = torch.rand(batch_size, 3, 448*2, 448*2).cuda()

    #output_dict1 = model1(image) # w/o bottleneck-linear
    output_dict2 = model2(image) # w/ bottleneck-linear

    #for k, v in output_dict1.items():
    #    print(k, v.shape)

    for k, v in output_dict2.items():
        print(k, v.shape)
