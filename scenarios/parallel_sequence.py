from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import create_parser
from script_manager.scenarios.common import create_launch_commands


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
    parser.add_argument(
        "--wandb_project_name", "-w", type=str, default="avalanche"
    )
    parser.add_argument("--gpus", "-g", type=int, nargs="*", default=-1)
    args = parser.parse_args()
    process_gpus(args)
    return args


def do_sequence(
    args,
    configs,
    default_parameters,
    launch_commands,
    script_file,
    test_parameters,
    wandb_project_name,
):
    if args.gpus != -1:
        launch_commands = create_launch_commands(
            launch_commands, args.gpus, len(configs)
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
