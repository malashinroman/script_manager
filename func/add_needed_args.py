import argparse
def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(',')]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(',')]


def append_needed_args(parser):
    default_args = parser.add_argument_group("default_args")
    default_args.add_argument("--tag", default="")
    default_args.add_argument("--output_dir", type=str, default='')
    default_args.add_argument("--wandb_project_name", type=str, default=None)
    default_args.add_argument("--tensorboard_folder", type=str, default="tensorboard")
    default_args.add_argument("--use_tensorboard", type=int, default=0)
