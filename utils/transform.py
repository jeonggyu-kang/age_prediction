from abc import ABC, abstractmethod
import sys
import random

import torch
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import numpy as np 

try:
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
except ImportError as e:
    print('{}: Please install albumentations by following command: pip install albumentations'.format(e))
    exit(1)
    

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self):
        pass

class ImageTransform(BaseTransform):
    def __call__(self, data):
        image = data['image']
        augmented = self.transform(image=image)
        image = augmented['image']

        data['image'] = image
        return data

    def __repr__(self):
        ret = ''
        ret += self.__class__.__name__
        
        if hasattr(self, 'p'):
            ret += ':' + str(self.p)
        if hasattr(self, 'width') and hasattr(self, 'height'):
            ret += '(HxW):' + str(self.height) + 'x' + str(self.width) 
        if hasattr(self, 'strength'):
            ret += ':' + str(self.strength)
            
        return ret

class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, list)

        self.transform = []
        _repr = []
        for transform in transforms:
            obj_type = transform.get('type')
            t = getattr(sys.modules[__name__], obj_type)(**transform)
            self.transform.append(t)
            _repr.append(t.__repr__())
        print(' - '.join(_repr))

    def __call__(self, data):
        for t in self.transform:
            data = t(data)
        
        return data

class ToTensor(ImageTransform):
    def __init__(self, **kwargs):
        self.transform = ToTensorV2()

class Normalize(ImageTransform):
    def __init__(self, **kwargs):
        mean = kwargs.get('mean')
        std = kwargs.get('std')

        self.transform = A.Normalize(mean=mean, std=std)

class GaussianNoise(ImageTransform):
    def __init__(self, **kwargs):
        self.p = kwargs.get('p')
        if self.p is None:
            self.p = 0.5

        self.strength = kwargs.get('strength')
        if self.strength is None:
            self.strength = 0.05

    def __call__(self, data):
        if random.random() < self.p:
            image = data['image']
            image += torch.randn(image.size()) * self.strength
            data['image'] = image
        
        return data

class HorizontalFlip(ImageTransform):
    def __init__(self, **kwargs):
        p = kwargs.get('p')
        if p is None:
            p = 0.5
        self.transform = A.HorizontalFlip(p=p)

class Rotation(ImageTransform):
    def __init__(self, **kwargs):
        self.p = kwargs.get('p')
        if self.p is None:
            self.p = 0.5

    def __call__(self, data):
        if random.random() < self.p:
            image = data['image']

            if random.randint(0,1) == 1:
                k = 1    
            else:
                k = 3
            image = np.rot90(image, k=k)
            data['image'] = image

        return data

class Resize(ImageTransform):
    def __init__(self, **kwargs):
        self.height = kwargs.get('height')
        self.width = kwargs.get('width')
        self.transform = A.Resize(height=self.height, width=self.width)

class ToGray(ImageTransform):
    def __init__(self, **kwargs):
        self.p = kwargs.get('p')
        if self.p is None:
            self.p = 0.5
        self.transform = A.ToGray(p=self.p)

class ColorJitter(ImageTransform):
    def __init__(self, **kwargs):
        self.transform = A.ColorJitter(p=kwargs.get('p'))

class RandomBrightnessContrast(ImageTransform):
    def __init__(self, **kwargs):
        self.transform = A.RandomBrightnessContrast(p=kwargs.get('p'))

class HueSaturationValue(ImageTransform):
    def __init__(self, **kwargs):
        self.transform = A.HueSaturationValue(p=kwargs.get('p'))

class CLAHE(ImageTransform):
    def __init__(self, **kwargs):
        self.transform = A.CLAHE(p=kwargs.get('p'))    


class Sharpness(ImageTransform):
    def __init__(self, **kwargs):

        self.p = kwargs.get('p')

        if self.p is None:
            self.p = 0.5 
        self.transform = A.Sharpen(p=1.0)      

    def __call__(self, data):
        
        if random.random() < self.p:
            image = data['image']
            image *= 255

            augmented = self.transform(image = image) # sharpen
            image = augmented['image'] / 255.0

            data['image'] = image
        
        return data
        

class Contrastive(ImageTransform):
    def __init__(self, **kwargs):
        self.p = kwargs.get('p')
        if self.p is None:
            self.p = 0.5

        self.w = kwargs.get('w')
        assert self.w is not None, 'W value must be provided.'

    def __call__(self, data):
        if random.random() < self.p:
            image = data['image'] # H x W x 1

            image = image ** self.w
            
            data['image'] = image
        return data

def build_transform_from_cfg(pipeline):
    transform = Compose(pipeline)
    
    return transform