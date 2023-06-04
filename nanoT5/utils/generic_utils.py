import os
from omegaconf import OmegaConf
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default = 'default',
        help = "Provide path to config.yaml if not provided default.yaml will be used"
    )
    args = parser.parse_args()
    return

def get_args(config_path = None):
  if config_path is None:
     config_path = parse_args()["config_path"]
     
  dir_path = os.path.dirname(__file__)
  if config_path == "default":
    config_path = os.path.join(dir_path, "../configs/default.yaml")
  elif config_path == "cluster":
    config_path = os.path.join(dir_path, "../configs/cluster.yaml")

  
  args = OmegaConf.load(config_path)

  if args.logging.wandb:
    if args.logging.wandb_credential_path == 'default':
        args.logging.wandb_credential_path = os.path.join(dir_path, "../configs/wandb_key.txt")
    with open(args.logging.wandb_credential_path, "r") as f:
      args.logging.wandb_key = f.read().strip()
  return args 