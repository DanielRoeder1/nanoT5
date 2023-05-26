from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time


from .utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)


import yaml
from types import SimpleNamespace
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

import os
def load_args(config_path = "default"):
  if config_path == "default":
    dir_path = os.path.dirname(__file__)
    config_path = config_path = os.path.join(dir_path, "configs/default.yaml")
  
  return load_config(config_path)

   

from .utils.data_scripts import get_dataset
from transformers import AutoTokenizer


def prepare_data():
  args = load_args()
  tokenizer = get_tokenizer(args)
  if args.model.knowledge_injection:
    know_tokenizer = AutoTokenizer.from_pretrained(args.model.know_enc_name) if args.model.know_enc_name != "T5" else tokenizer
  dataset = get_dataset(args.data.data_dir, tokenizer, know_tokenizer=know_tokenizer)
  print(dataset)
  return dataset

if __name__ == "__main__":
  dataset = prepare_data()

  dataset.save_to_disk("/content/sample_data/hf_data")
