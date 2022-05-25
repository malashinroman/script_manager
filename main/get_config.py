import argparse

from script_manager.func.add_needed_args import append_needed_args
from script_manager.func.wandb_logger import init_wandb_logger, update_args_from_wandb
from local_config import WANDB_LOGIN

def get_args():
    parser = argparse.ArgumentParser(description="Sample_Project")
    append_needed_args(parser)

    parser.add_argument("--number_of_iterations", type=int, default=1)
    parser.add_argument("--imagenet_path", type=str, default='')
    parser.add_argument("--prev_result", type=str, default=None)


    return parser.parse_args()


def get_args_and_wandb():
    args = get_args()
    init_wandb_logger(args, WANDB_LOGIN)
    args = update_args_from_wandb(args)
    return args

# def update_args_from_wandb(program_args):
#     init_wandb_logger(program_args, WANDB_LOGIN)
#     args = update_args_from_wandb(program_args)
#     return args