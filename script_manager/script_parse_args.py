import argparse
def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(',')]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(',')]


parser = argparse.ArgumentParser(description="TargetRecogniton")
parser.add_argument("--not_test", action="store_true")
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--enable_wandb", action="store_true")
parser.add_argument("--parallel_num", type=int, default=1)
parser.add_argument("--configs2run", type=str2intlist, default=None)
parser.add_argument("--cwd", type=str, default=None)

def get_script_args():
    args = parser.parse_args()
    return args