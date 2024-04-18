import argparse


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(",")]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(",")]


def create_parser():
    parser = argparse.ArgumentParser(description="TargetRecogniton")
    parser.add_argument(
        "-n",
        "--not_test",
        action="store_true",
        help="if set to True, than test_parameters will be ignored",
    )
    parser.add_argument(
        "--sleep",
        type=str,
        default="0",
        help="sleep time before starting the script (10s - 10 seconds, 10m -10 minutes, 10h - 10 hours)",
    )
    parser.add_argument(
        "-e",
        "--enable_wandb",
        action="store_true",
        help="if set to True, than wandb will be enabled",
    )
    parser.add_argument(
        "-p",
        "--parallel_num",
        type=int,
        default=1,
        help="specifies number of processes in parallel run, if set to 2, than the script will be run in parallel with 2 processes",
    )
    parser.add_argument("-c", "--configs2run", type=int, nargs="*",
                        default=None, help="list of indexes of scripts to run (for debug)")
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    return parser


def get_script_args():
    parser = create_parser()
    args = parser.parse_args()
    return args
