import json
import os
import pprint


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(",")]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(",")]


def append_needed_args(parser):
    default_args = parser.add_argument_group("default_args")
    default_args.add_argument("--tag", default="")
    default_args.add_argument("--output_dir", type=str, default="")
    default_args.add_argument("--wandb_project_name", type=str, default=None)
    default_args.add_argument("--tensorboard_folder", type=str, default="tensorboard")
    default_args.add_argument("--use_tensorboard", type=int, default=0)
    default_args.add_argument("--kill_concurrent_folders", type=int, default=0)


def dump_args_as_default_paraemeters(args):
    pprint.pprint(args.__dict__)


def smart_parse_args(parser):
    append_needed_args(parser)
    args = parser.parse_args()
    if args.wandb_project_name is not None:
        from script_manager.func.wandb_logger import prepare_wandb

        args = prepare_wandb(args)

    os.makedirs(args.output_dir, exist_ok=True)
    param_path = os.path.join(args.output_dir, "run_params.json")

    with open(param_path, "w") as fp:
        if args.wandb_project_name is not None:
            json.dump(args._items, fp, indent=4, sort_keys=True)
        else:
            json.dump(args.__dict__, fp, indent=4, sort_keys=True)

    return args
