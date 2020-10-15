from omegaconf.dictconfig import DictConfig
from peddet.transforms import TrainTransforms, ValTransforms
from peddet.data import PennFudanDataModule
import pytest


@pytest.mark.timeout(2)
def test_datamodule_init(sample_config: DictConfig):
    "Assert no exception raised while initializing lightning datamodule"
    cfg = sample_config
    train_transforms = TrainTransforms(cfg.aug)
    val_transforms = ValTransforms(cfg.aug)
    datamodule = PennFudanDataModule(cfg, train_transforms, val_transforms)
    datamodule.setup(stage="fit")


@pytest.mark.timeout(6)
def test_train_dataloader(sample_config: DictConfig):
    """Assert no exception raised while instantiating train dataloder
    and reading first batch
    """
    cfg = sample_config
    train_transforms = TrainTransforms(cfg.aug)
    val_transforms = ValTransforms(cfg.aug)
    datamodule = PennFudanDataModule(cfg, train_transforms, val_transforms)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    images, targets = next(iter(train_loader))


@pytest.mark.timeout(6)
def test_val_dataloader(sample_config: DictConfig):
    """Assert no exception raised while instantiating val dataloder
    and reading first batch
    """
    cfg = sample_config
    train_transforms = TrainTransforms(cfg.aug)
    val_transforms = ValTransforms(cfg.aug)
    datamodule = PennFudanDataModule(cfg, train_transforms, val_transforms)
    datamodule.setup(stage="test")
    val_loader = datamodule.val_dataloader()
    images, targets = next(iter(val_loader))
