import os
from omegaconf import OmegaConf

def get_args(config_path = "default"):
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