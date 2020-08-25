from hydra.core.config_store import ConfigStore
from hydra.types import TargetConf
from dataclasses import dataclass
from functools import partial
from typing import List
import re

cs = ConfigStore.instance()
ALBU_TRANSFORMS: str = "albumentations.augmentations.transforms"

class HueSatConf(TargetConf):
    _target_: str = f"{ALBU_TRANSFORMS}.HueSaturationValue"
    hue_shift_limit: float = 0.2
    sat_shift_limit: float = 0.2
    val_shift_limit: float = 0.2
    p: float =0.9

class BrightnessContrastConf(TargetConf):
    _target_: str = f"{ALBU_TRANSFORMS}.RandomBrightnessContrast"
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    p: float = 0.9

class ToGray(TargetConf):
    _target_: str = f"{ALBU_TRANSFORMS}.ToGray"
    p: float = 0.01

class FlipConf(TargetConf):
    _target_: str = f"{ALBU_TRANSFORMS}.Flip"
    always_apply: bool = True

class CutoutConf(TargetConf):
    _target_: str = f"{ALBU_TRANSFORMS}.Cutout"
    num_holes: int = 8
    max_h_size: int = 64
    max_w_size: int = 64
    fill_value: int = 0
    p: float = 0.5

class ToTensorConf(TargetConf):
    _target_: str = f"{ALBU_TRANSFORMS}.ToTensorV2"
    p: float = 1.0

class BboxParamsConf(TargetConf):
    _target_: str = f"albumentations.core.composition.BboxParams"
    format: str ='coco'
    min_area: int = 0
    min_visibility: int = 0
    label_fields: List[str] = ['labels']


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name):
    "Convert CamelCase to snake_case"
    s1   = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def formulate_name(classname: str):
    snake_case = camel2snake(classname)
    name = re.sub('_conf','',snake_case)
    return name


configs: List[type] = [FlipConf,BrightnessContrastConf,CutoutConf,ToTensorConf,ToGray,HueSatConf,BboxParamsConf]

for conf in configs:
    name = formulate_name(conf.__name__)
    cs.store(group='aug',name=name,node=conf)

# if __name__ == "__main__":
#     for conf in configs:
#         name = formulate_name(conf.__name__)
#         print(name)
#
#     Output:
#     flip
#     brightness_contrast
#     cutout
#     to_tensor
#     to_gray
#     hue_sat
#     bbox_params
