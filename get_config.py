import argparse
from func.wandb_logger import init_wandb_logger, update_args_from_wandb
from local_config import WANDB_LOGIN

def get_args():
    parser = argparse.ArgumentParser(description="Sample_Project")
    parser.add_argument("--tag", default="")
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--wandb_project_name", type=str, default=None)

    parser.add_argument("--number_of_iterations", type=int, default=1)
    parser.add_argument("--imagenet_path", type=str, default='')
    parser.add_argument("--prev_result", type=str, default=None)


    return parser.parse_args()


def get_args_and_wandb():
    args = get_args()
    init_wandb_logger(args, WANDB_LOGIN)
    args = update_args_from_wandb(args)
    return args