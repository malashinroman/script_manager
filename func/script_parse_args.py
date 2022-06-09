import argparse


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(",")]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(",")]


def get_script_args():
    parser = argparse.ArgumentParser(description="TargetRecogniton")
    parser.add_argument("--not_test", action="store_true")
    parser.add_argument("--sleep", type=str, default="0")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--configs2run", type=str2intlist, default=None)
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    return args
