import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import conf.register_config

@hydra.main(config_path="conf",config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg.aug))

if __name__ == "__main__":
    main()