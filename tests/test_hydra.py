import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.experimental import initialize, compose
import pytest
from albumentations.augmentations.transforms import Flip
from albumentations.core.composition import BboxParams

def test_hydra_aug_instantiate() -> None:
    "Assert hydra instantiation works"
    with initialize(config_path='../conf'):
        cfg = compose(config_name='config')
        obj = instantiate(cfg.aug.flip)
        assert isinstance(obj, Flip)

def test_hydra_bbox_params_instantiate() -> None:
    "Assert hydra instantiation works"
    with initialize(config_path='../conf'):
        cfg = compose(config_name='config')
        obj = instantiate(cfg.aug.bbox_params)
        assert isinstance(obj, BboxParams)

def test_hydra_init() -> None:
    "Assert no exception is raised while initializing config"
    with initialize(config_path='../conf'):
        cfg = compose(config_name='config')
