import timm

import torch
import torch.nn as nn


def get_eleme_num(model, x):
    y = model(x)
    _, C, H, W = y.shape
    return C*H*W


class CustomizationNet(nn.Module):
    def __init__(self, num_classes, network_name, image_size):
        super(CustomizationNet, self).__init__()

        self.feature_extractor = timm.create_model(network_name, pretrained = False)
        self.feature_extractor.reset_classifier(0, '')

        num_elems = get_eleme_num(self.feature_extractor, torch.randn(*image_size))

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_elems, num_classes)
        )

    def forward(self, x):
        feature = self.feature_extractor(x)

        pred = self.linear(feature)

        return pred

if __name__ == '__main__':


    model_names = timm.list_models('*hr*', pretrained=True)
    #print(len(model_names))
    print(model_names)

    for model_name in model_names:
        print(model_name)
        model = CustomizationNet(num_classes = 4, network_name = model_name, image_size=(1,3,448,448))

        image = torch.rand(2, 3, 448, 448)
        pred = model(image)
        print(pred)


    '''
    model = CustomizationNet(num_classes = 4, network_name = 'mobilenetv2_050')
    x = torch.randn(2,3,224,224) # N x C x H x W
    pred = model(x)
    print (pred.shape)
    '''

    