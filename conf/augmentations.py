from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf
from dataclasses import dataclass, field
from functools import partial
from typing import List
import re

# cs = ConfigStore.instance()
ALBU_TRANSFORMS: str = "albumentations.augmentations.transforms"

################################################################
## Start of augmentation configs
################################################################

@dataclass
class HueSatConf:
    _target_: str = f"{ALBU_TRANSFORMS}.HueSaturationValue"
    hue_shift_limit: float = 0.2
    sat_shift_limit: float = 0.2
    val_shift_limit: float = 0.2
    p: float =0.9

@dataclass
class BrightnessContrastConf:
    _target_: str = f"{ALBU_TRANSFORMS}.RandomBrightnessContrast"
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    p: float = 0.9

@dataclass
class ToGray:
    _target_: str = f"{ALBU_TRANSFORMS}.ToGray"
    p: float = 0.01

@dataclass
class FlipConf:
    _target_: str = f"{ALBU_TRANSFORMS}.Flip"
    always_apply: bool = True

@dataclass
class CutoutConf:
    _target_: str = f"{ALBU_TRANSFORMS}.Cutout"
    num_holes: int = 8
    max_h_size: int = 64
    max_w_size: int = 64
    fill_value: int = 0
    p: float = 0.5

@dataclass
class ToTensorConf:
    _target_: str = f"{ALBU_TRANSFORMS}.ToTensorV2"
    p: float = 1.0

def default_field(obj):
    return field(default_factory=lambda: obj)

@dataclass
class BboxParamsConf:
    _target_: str = f"albumentations.core.composition.BboxParams"
    format: str ='coco'
    min_area: int = 0
    min_visibility: int = 0
    label_fields: List[str] = default_field(['labels'])

################################################################
## End of augmentation configs
################################################################

configs: List[type] = [FlipConf,BrightnessContrastConf,CutoutConf,ToTensorConf,ToGray,HueSatConf,BboxParamsConf]

# if __name__ == "__main__":
#     for conf in configs:
#         name = formulate_name(conf.__name__)
#         print(name)
#     Output:
#     flip
#     brightness_contrast
#     cutout
#     to_tensor
#     to_gray
#     hue_sat
#     bbox_params
