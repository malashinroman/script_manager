import json
import os
from pathlib import PosixPath
import pprint
import pkg_resources as pkg

def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(",")]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(",")]


def append_needed_args(parser):
    """
    Append the needed arguments to the parser.
    """
    default_args = parser.add_argument_group("default_args")
    default_args.add_argument("--tag", default="")
    default_args.add_argument("--output_dir", type=str, default="")
    default_args.add_argument("--wandb_project_name", type=str, default=None)
    default_args.add_argument("--tensorboard_folder", type=str, default="tensorboard")
    default_args.add_argument("--use_tensorboard", type=int, default=0)
    default_args.add_argument("--kill_concurrent_folders", type=int, default=0)
    default_args.add_argument("--random_seed", type=int, default=None)

def dump_args_as_default_paraemeters(args):
    pprint.pprint(args.__dict__)

def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum) 
    s = f'WARNING ⚠️ {name}{minimum} is required by script_manager, but {name}{current} is currently installed' 
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        print('s')
    return result

def set_random_seed(random_seed, deterministic=False):
        import numpy as np
        import torch
        import random
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        # FIXME: causes error, when adding CUBLAS_... env variable
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True
        if deterministic and check_version(torch.__version__, '1.12.0'):  # https://github.com/ultralytics/yolov5/pull/8213
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)

def convert_args_for_json(args):
    args_dict = args.__dict__

    json_dict = {}
    for k,v in args_dict.items():
        if isinstance(v, PosixPath):
            v_save = str(v)
        else:
            v_save = v
        json_dict[k] = v_save
    return json_dict

def smart_parse_args(parser):
    """
    Function to call instead of parser.parse_args().
    append needed args,
    prepares wandb_project
    parses args
    """
    append_needed_args(parser)
    args = parser.parse_args()
    from script_manager.func.wandb_logger import prepare_wandb

    args = prepare_wandb(args)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # create output dir if specified
    # it may not be specified if we don't use do_everything
    if len(args.output_dir) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        param_path = os.path.join(args.output_dir, "run_params.json")

        with open(param_path, "w") as fp:
            json_dict = convert_args_for_json(args)
            json.dump(json_dict, fp, indent=4, sort_keys=True)

    return args
