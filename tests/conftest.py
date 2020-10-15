from typing import Union
from hydra._internal.hydra import Hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.experimental import initialize, compose
import pytest
from omegaconf import DictConfig


@pytest.fixture(scope="package")
def hydra_instance() -> Union[Hydra, GlobalHydra]:
    "Provide Hydra/GlobalHydra instance for compose"
    if HydraConfig.initialized():
        yield GlobalHydra.instance()
    hydra_init = initialize(config_path="../peddet/conf")
    yield hydra_init
    GlobalHydra.instance().clear()


def test_hydra_init(hydra_instance):
    "Assert no exception raised while initializing config"
    cfg = compose(config_name="config")
    assert isinstance(cfg, DictConfig)


@pytest.fixture(scope="package")
def default_config(hydra_instance) -> DictConfig:
    "Provide DictConfig composed with defaults"
    yield compose(config_name="config")


def test_hydra_override(hydra_instance):
    "Assert no exception raised while initializing config"
    cfg = compose(config_name="config", overrides=["data=sample"])
    assert isinstance(cfg, DictConfig)


@pytest.fixture(scope="package")
def sample_config(hydra_instance) -> DictConfig:
    "Provide DictConfig composed with sample dataset"
    yield compose(config_name="config", overrides=["data=sample"])