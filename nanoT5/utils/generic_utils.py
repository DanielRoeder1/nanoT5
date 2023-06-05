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
    parser.add_argument(
        "--mode",
        type=str,
        default = None,
        help = "Overwrite model.mode in config.yaml"
    )
    args = parser.parse_args()
    return args

def get_args(config_path = None, load_cmd =  True):
  if load_cmd:
    cmd_args = parse_args()
  if config_path is None:
     config_path = cmd_args.config_path
     
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

  if load_cmd and cmd_args.mode is not None:
     args.model.mode = cmd_args.mode
  return args 


if __name__ == "__main__":
  args = get_args()
  print(args)