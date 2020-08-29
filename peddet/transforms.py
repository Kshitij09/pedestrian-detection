from omegaconf import DictConfig
from hydra.utils import instantiate
from albumentations.core.composition import Compose, OneOf

class TrainTransforms(object):

    def __init__(self, aug: DictConfig):
        color_tfms = [aug.tfms.brightness_contrast,aug.tfms.hue_sat]
        color_transform = OneOf([instantiate(tfm) for tfm in color_tfms],p=0.9)
        tfm_list = [color_transform] + [instantiate(tfm) for tfm in aug.tfms.values() if tfm not in color_tfms]
        self.transforms = Compose(tfm_list, bbox_params=instantiate(aug.bbox_params))
    
    def __call__(self, *args, **kwargs): return self.transforms(*args,**kwargs)

class ValTransforms(object):
    def __init__(self, aug: DictConfig):
        self.transforms = Compose(
            [instantiate(aug.tfms.to_tensor)],
            bbox_params=instantiate(aug.bbox_params)
        )
    
    def __call__(self, *args, **kwargs): return self.transforms(*args,**kwargs)

