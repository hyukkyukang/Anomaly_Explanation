import os

import hydra
import omegaconf
from omegaconf import OmegaConf

# Load default config
file_dir_path = os.path.dirname(os.path.abspath(__file__))
default_config = OmegaConf.load(os.path.join(file_dir_path, "default.yaml"))


def dump_config(config: OmegaConf, path: str) -> None:
    """Dump config to console and save to file."""
    print(OmegaConf.to_yaml(config))
    OmegaConf.save(config, path)


def load_config(path: str) -> OmegaConf:
    """Load config from file."""
    config = OmegaConf.load(path)
    return config


@hydra.main(
    config_path="../configs/",
    config_name="base",
    version_base="1.3",
)
def configs(cfg) -> omegaconf.DictConfig:
    return cfg
