import itertools
import os
import random

from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import create_parser


def set_gpus(args, gpus):
    print(f"Setting custom GPU configuration, GPUs: {gpus}")
    args.gpus = gpus
    process_gpus(args)
    return args


def set_parallel_num(args, parallel_num):
    print(f"Setting custom number of parallel processes: {parallel_num}")
    args.parallel_num = parallel_num
    return args


def process_gpus(args):
    if isinstance(args.gpus, list):
        if args.parallel_num > 1:
            print("Using multiple threads")
        else:
            args.gpus = [str(args.gpus)[1:-1].replace(" ", "")]


def parse_args():
    parser = create_parser()
    parser.add_argument("--ssd", type=int, default=0)
    parser.add_argument("--processes", type=int, default=0)
    parser.add_argument("--wandb_project_name", "-w", type=str, default="avalanche")
    parser.add_argument("--gpus", "-g", type=int, nargs="*", default=-1)
    args = parser.parse_args()
    process_gpus(args)
    return args


def create_configs(
    variable_parameters: dict,
    random_shuffle=True,
    dict2tag_func=None,
    base_tag="",
):
    if dict2tag_func is None:
        dict2tag_func = (lambda x: "_".join(f"{k}-{v}" for k, v in x.items()),)

    # weights and biases project name
    combinations = list(itertools.product(*variable_parameters.values()))

    parameter_configurations = []

    for combination in combinations:
        parameter_configuration = dict(zip(variable_parameters.keys(), combination))
        combination_str = dict2tag_func(parameter_configuration)

        tag = f"{base_tag}_{combination_str}"
        parameter_configuration["tag"] = tag
        parameter_configurations.append(parameter_configuration)

    configs = []
    for parameter_configuration in parameter_configurations:
        configs.append([parameter_configuration, None])

    if random_shuffle:
        random.shuffle(configs)

    return configs


def create_launch_commands(main_script, gpus2use, configs_number):
    launch_commands = []
    for index in range(configs_number):
        if isinstance(gpus2use, list):
            gpu = gpus2use[index % len(gpus2use)]
            launch_commands.append(f"CUDA_VISIBLE_DEVICES={gpu} python {main_script}")
        else:
            # isinstance(gpus2use, int):
            launch_commands.append(main_script)
    return launch_commands


def create_configs_and_launch_commands(
    main_script, variable_parameters, gpus2use, dict2tag_func, base_tag
):
    configs = create_configs(
        variable_parameters, dict2tag_func=dict2tag_func, base_tag=base_tag
    )
    launch_commands = create_launch_commands(main_script, gpus2use, len(configs))
    return configs, launch_commands


def do_parallel_parameter_search(
    args,
    default_parameters,
    main_script,
    script_file,
    test_parameters,
    variable_parameters,
    wandb_project_name,
    dict2tag_func=None,
):

    base_tag = os.path.split(script_file)[-1].split(".")[0]
    configs, launch_commands = create_configs_and_launch_commands(
        main_script,
        variable_parameters,
        gpus2use=args.gpus,
        dict2tag_func=dict2tag_func,
        base_tag=base_tag,
    )
    do_everything(
        default_parameters=default_parameters,
        configs=configs,
        extra_folder_keys=[],
        appendix_keys=[],
        main_script=launch_commands,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
        script_file=script_file,
        args=args,
    )
