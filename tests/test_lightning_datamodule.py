from peddet.transforms import TrainTransforms, ValTransforms
from peddet.data import PennFudanDataModule
from hydra.experimental import initialize, compose

def test_datamodule_init():
    "Assert no exception raised while initializing lightning datamodule"
    with initialize(config_path='../peddet/conf'):
        cfg = compose(config_name='config')
        train_transforms = TrainTransforms(cfg.aug)
        val_transforms = ValTransforms(cfg.aug)
        datamodule = PennFudanDataModule(cfg,train_transforms,val_transforms)
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        image,target,image_id = next(iter(train_loader))

def test_train_dataloader():
    """Assert no exception raised while instantiating train dataloder
       and reading first batch
    """
    with initialize(config_path='../peddet/conf'):
        cfg = compose(config_name='config')
        train_transforms = TrainTransforms(cfg.aug)
        val_transforms = ValTransforms(cfg.aug)
        datamodule = PennFudanDataModule(cfg,train_transforms,val_transforms)
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        image,target,image_id = next(iter(train_loader))

def test_val_dataloader():
    """Assert no exception raised while instantiating train dataloder
       and reading first batch
    """
    with initialize(config_path='../peddet/conf'):
        cfg = compose(config_name='config')
        train_transforms = TrainTransforms(cfg.aug)
        val_transforms = ValTransforms(cfg.aug)
        datamodule = PennFudanDataModule(cfg,train_transforms,val_transforms)
        datamodule.setup(stage='fit')
        val_loader = datamodule.val_dataloader()
        image,target,image_id = next(iter(val_loader))
