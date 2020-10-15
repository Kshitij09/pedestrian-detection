from omegaconf import DictConfig
from hydra.utils import instantiate
from albumentations.core.composition import Compose
import albumentations as A
import warnings

# Ignoring Tensorboard DeprecationWarning for tensorboard's FieldDescriptor()
warnings.filterwarnings(action="ignore", module="tensorboard.compat.proto")
# Ignoring DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc'
warnings.filterwarnings(
    action="ignore", module="pytorch_lightning.overrides.data_parallel"
)


def test_hydra_aug_instantiate(default_config: DictConfig):
    "Assert hydra instantiation works"
    cfg = default_config
    obj = instantiate(cfg.aug.tfms.to_gray)
    assert isinstance(obj, A.ToGray)


def test_albumentations_compose(default_config: DictConfig):
    "Assert no exception raised while composing albumentations"
    aug = default_config.aug
    color_tfms = [aug.tfms.brightness_contrast, aug.tfms.hue_sat]
    color_transform = A.OneOf([instantiate(tfm) for tfm in color_tfms], p=0.9)
    transforms = Compose(
        [instantiate(tfm) for tfm in aug.tfms.values() if tfm not in color_tfms],
        bbox_params=instantiate(aug.bbox_params),
    )