import os
import sys
from copy import deepcopy

sys.path.append(".")
from func.make_command import make_command2
from local_config import IMAGENET_PATH

from pathlib import Path

from func.run_series_of_experiments import run_async
from func.script_parse_args import get_script_args


def get_main_script():
    main_script = os.path.join("hello_imagenet.py")
    return main_script


def get_wandb_project_name():
    return "hello_imagenet"


def get_name_keys():
    # this parameters will be logged as appendix of the directory name
    folder_keys = [
        os.path.basename(os.path.dirname(os.path.dirname(__file__))),
        os.path.basename(os.path.dirname(__file__)),
        os.path.basename(__file__).split(".")[0],
    ]

    # this parameters will be logged as as subfolders in directory path
    appendix_keys = ["tag"]

    return folder_keys, appendix_keys


default_parameters = {
    'imagenet_path': IMAGENET_PATH
}


def set_configs():
    configs = []
    config1 = \
        {
            'number_of_iterations': 1,
            'tag': "num_1_",
        }

    configs.append([config1, None])
    config2 = \
        {
            'number_of_iterations': 10,
            'tag': "num_10_"
        }
    configs.append([config2, "prev_result"])

    return configs


def get_test_update_dict(config):
    d = {'number_of_iterations': 0}
    if "wandb_project_name" in config:
        if config["wandb_project_name"] is not None:
            config["wandb_project_name"] = "debug_" + config["wandb_project_name"]
    return d


def get_name_keys():
    # this parameters will be logged as appendix of the directory name
    folder_keys = []

    # this parameters will be logged as as subfolders in directory path
    appendix_keys = ["tag"]

    return folder_keys, appendix_keys


def get_cwd(args):
    if args.cwd is None:
        cwd = str(path.parent.parent.parent.parent.parent.absolute())
    else:
        cwd = args.cwd
    print({'cwd': cwd})
    return cwd


def configs2cmds(full_configs, default_parameters, main_script, args, folder_keys, appedix_keys):
    configs = [f[0] for f in full_configs]
    uof = [f[1] for f in full_configs]
    assert (len(uof) == len(configs))

    # folder_keys, appedix_keys = get_name_keys()
    run_list = []
    for data_configuration, output_forward_key in zip(configs, uof):
        configuration_dict = deepcopy(default_parameters)
        configuration_dict.update(data_configuration)
        if output_forward_key is not None:
            if len(output_forward_key) > 0:
                assert (pref_step_output is not None)
                configuration_dict[output_forward_key] = pref_step_output
        if not args.not_test:
            test_parameters = get_test_update_dict(configuration_dict)
            configuration_dict.update(test_parameters)
            print("WARNING: test mode is activated")
        if args.enable_wandb:
            configuration_dict['wandb_project_name'] = get_wandb_project_name()
        else:
            print("WARNING: no wandb logs are enabled")

        cmd0, pref_step_output = make_command2(configuration_dict, main_script, folder_keys, appedix_keys)
        # pref_step_output = output0
        run_list.append(cmd0)

    if args.configs2run is not None:
        final_run_list = [run_list[i] for i in args.configs2run]
    else:
        final_run_list = run_list

    final_run_list.insert(0, f'sleep {args.sleep}')
    return final_run_list


def run_cmds(run_list, cwd, args):
    run_async(
        run_list,
        parallel_num=args.parallel_num,
        cwd=cwd,
    )


if __name__ == '__main__':
    args = get_script_args()

    path = Path(__file__)
    work_dir = get_cwd(args)
    configs = set_configs()

    folder_keys = [
        os.path.basename(os.path.dirname(os.path.dirname(__file__))),
        os.path.basename(os.path.dirname(__file__)),
        os.path.basename(__file__).split(".")[0],
    ]
    extra_folder_keys, appendix_keys = get_name_keys()
    folder_keys += extra_folder_keys
    run_list = configs2cmds(configs, default_parameters, get_main_script(), args, folder_keys,
                            appedix_keys=appendix_keys)
    run_cmds(run_list, work_dir, args)
