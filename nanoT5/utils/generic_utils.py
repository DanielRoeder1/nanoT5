import yaml
from types import SimpleNamespace
import os

# YAML -> SimpleNamespace
# from: https://gist.github.com/jdthorpe/313cafc6bdaedfbc7d8c32fcef799fbf
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=Loader)
    return config

class Loader(yaml.Loader):
    pass

def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))

Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)

def get_args(config_path = "default"):
  if config_path == "default":
    dir_path = os.path.dirname(__file__)
    config_path = config_path = os.path.join(dir_path, "../configs/default.yaml")
  
  return load_config(config_path)
