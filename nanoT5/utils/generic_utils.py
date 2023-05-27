import os
from omegaconf import OmegaConf

def get_args(config_path = "default"):
  if config_path == "default":
    dir_path = os.path.dirname(__file__)
    config_path = os.path.join(dir_path, "../configs/default.yaml")
  
  return OmegaConf.load(config_path)



