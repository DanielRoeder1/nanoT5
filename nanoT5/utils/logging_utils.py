from collections import defaultdict

from accelerate.logging import get_logger
import logging
import datasets
import transformers
import os

import wandb


class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        self.reset()

        return averaged_stats


class Logger:
    def __init__(self, args, accelerator):
        self.logger = get_logger('Main')

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(accelerator.state, main_process_only=False)
        self.logger.info(f'Working directory is {os.getcwd()}')

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        self.setup_wandb(args)

    def setup_wandb(self, args):
        if args.logging.wandb:
            wandb.login(key = args.logging.wandb_key)
            wandb.init(args.logging.project_name, config=args.__dict__)

    def log_stats(self, stats, step, args, prefix=''):
        if args.logging.wandb:
            for k, v in stats.items():
                wandb.log({f'{prefix}{k}': v}, step=step)

        msg_start = f'[{prefix[:-1]}] Step {step} out of {args.optim.total_steps}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.3f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)

    def finish(self):
        wandb.finish()
