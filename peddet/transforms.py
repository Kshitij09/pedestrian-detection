from albumentations.core.composition import Compose
import albumentations as A
from dataclasses import dataclass
from omegaconf import DictConfig

class TrainTransforms(object):
    """Training transforms for image and bounding boxes.
       Current implementation is based on albumentations transforms
    """
    def __init__(self, aug: DictConfig):
        self.transforms = A.Compose([A.OneOf([\
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)
                ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),                      
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0)
        ],bbox_params=A.BboxParams(format='coco',min_area=0, min_visibility=0,label_fields=['labels'])
    )