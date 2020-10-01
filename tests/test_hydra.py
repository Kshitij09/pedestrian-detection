import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.experimental import initialize, compose
import pytest
from albumentations.core.composition import BboxParams
from albumentations.core.composition import Compose
import albumentations as A


@pytest.mark.timeout(30)
def test_hydra_aug_instantiate():
    "Assert hydra instantiation works"
    with initialize(config_path="../peddet/conf"):
        cfg = compose(config_name="config")
        obj = instantiate(cfg.aug.tfms.to_gray)
        assert isinstance(obj, A.ToGray)


@pytest.mark.timeout(30)
def test_hydra_init():
    "Assert no exception raised while initializing config"
    with initialize(config_path="../peddet/conf"):
        cfg = compose(config_name="config")


@pytest.mark.timeout(30)
def test_albumentations_compose():
    "Assert no exception raised while composing albumentations"
    with initialize(config_path="../peddet/conf"):
        aug = compose(config_name="config").aug
        color_tfms = [aug.tfms.brightness_contrast, aug.tfms.hue_sat]
        color_transform = A.OneOf([instantiate(tfm) for tfm in color_tfms], p=0.9)
        transforms = Compose(
            [instantiate(tfm) for tfm in aug.tfms.values() if tfm not in color_tfms],
            bbox_params=instantiate(aug.bbox_params),
        )
