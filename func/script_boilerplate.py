"""Typical functionality of the master script"""

import os
import time
from copy import deepcopy
from pathlib import Path

from script_manager.func.make_command import make_command2
from script_manager.func.run_series_of_experiments import run_async
from script_manager.func.script_parse_args import get_script_args


def update_with_test_dict(full_config, test_dict):
    # d = {'config-file': 'configs/trainers/da/sparse_dael/mini_domainnet_test.yaml'}
    full_config.update(test_dict)
    if "wandb_project_name" in full_config:
        if full_config["wandb_project_name"] is not None:
            full_config["wandb_project_name"] = (
                "debug_" + full_config["wandb_project_name"]
            )
    if "tag" in full_config:
        full_config["tag"] = "test_" + full_config["tag"]

    return full_config


def get_cwd(args, file_path):
    """Get the current working directory
    determining where to run the experiments
    By default assumes that the directory contains the script_manager"""

    path = Path(file_path)
    cwd = None
    if args.cwd is None:
        script_manager_found = False
        c_path = path
        while not script_manager_found:
            sm_path = os.path.join(str(c_path), "script_manager")
            if os.path.exists(sm_path):
                script_manager_found = True
                cwd = str(c_path)

            c_path = c_path.parent
        assert script_manager_found
        # cwd = str(path.parent.parent.parent.parent.parent.absolute())
    else:
        cwd = args.cwd
    print({"cwd": cwd})
    return cwd


def get_key_val_from_key(key_word):
    val = "parameter_without_value"
    key = key_word
    if "=" in key_word:
        kv_list = key_word.split("=")
        key = kv_list[0]
        val = kv_list[1]
    return key, val


def update_dict_from_opt_args(opt_args):
    ret_d = {}
    current_key = None
    for el in opt_args:
        if not (len(el)) < 1 and el[0] == "-":
            if current_key is not None:
                key, val = get_key_val_from_key(current_key)
                ret_d[key] = val
            current_key = el.strip("-")
        else:
            assert current_key is not None
            val = el
            ret_d[current_key] = val
            current_key = None

    if current_key is not None:
        key, val = get_key_val_from_key(current_key)
        ret_d[key] = val

    return ret_d


def configs2cmds(
    full_configs,
    default_parameters,
    main_script_container,
    args,
    folder_keys,
    appedix_keys,
    test_parameters,
    wandb_project_name,
    work_dir,
):
    """
    Convert the configs to commands
    Args:
    full_configs          : list of (dict, path), each dict is a config, path tunnels output_dir (use None by default)"
    default_parameters    : dict of default parameters
    main_script_container : str, path to the main script
    args                  : argparse.Namespace, arguments passed to the master script
    folder_keys           : list of str, keys to use for the folder name
    appedix_keys          : list of str, keys to use for the appendix name
    test_parameters       : dict of parameters to use for testing
    wandb_project_name    : str, name of the wandb project
    work_dir              : str, path where to run the experiments
    """
    configs = [f[0] for f in full_configs]
    uof = [f[1] for f in full_configs]
    assert len(uof) == len(configs)

    # folder_keys, appedix_keys = get_name_keys()
    run_list = []
    args_update = update_dict_from_opt_args(args.opts[1:])

    pref_step_output = None
    for conf_index, config_data in enumerate(zip(configs, uof)):
        if (
            isinstance(config_data, tuple)
            or isinstance(config_data, list)
            or len(config_data) > 1
        ):
            (data_configuration, output_forward_key) = config_data
        else:
            data_configuration = config_data
            output_forward_key = None

        if isinstance(data_configuration, str):
            the_cmd = data_configuration
            cmd_params = the_cmd.split(" ")
            if cmd_params[0] == "sleepnow":
                # we want to sleep before creating next run
                time.sleep(int(cmd_params[1]))
            else:
                run_list.append(data_configuration)
            continue
        configuration_dict = deepcopy(default_parameters)
        configuration_dict.update(data_configuration)
        if output_forward_key is not None:
            if len(output_forward_key) > 0:
                assert (
                    pref_step_output is not None
                ), "Trying to pass output of the script previous to 0"
                if type(output_forward_key) is dict:
                    key = list(output_forward_key.keys())[0]
                    path_format = list(output_forward_key.values())[0]
                    configuration_dict[key] = path_format.format(pref_step_output)
                else:
                    configuration_dict[output_forward_key] = pref_step_output
        if args.enable_wandb:
            configuration_dict["wandb_project_name"] = wandb_project_name
        if not args.not_test:
            # test_parameters = get_test_update_dict(configuration_dict)
            # configuration_dict.update(test_parameters)
            update_with_test_dict(configuration_dict, test_dict=test_parameters)
            print("WARNING: test mode is activated")
        # else:
        #     print("WARNING: no wandb logs are enabled")

        configuration_dict.update(args_update)
        if type(main_script_container) is list:
            main_script = main_script_container[conf_index]
        else:
            main_script = main_script_container

        cmd0, pref_step_output = make_command2(
            configuration_dict,
            main_script,
            folder_keys,
            appedix_keys,
            work_dir=work_dir,
        )
        run_list.append(cmd0)

    if args.configs2run is not None:
        final_run_list = [run_list[i] for i in args.configs2run]
    else:
        final_run_list = run_list

    if "h" in args.sleep:
        sleep_seconds = int(args.sleep.strip("h")) * 60 * 60
    elif "m" in args.sleep:
        sleep_seconds = int(args.sleep.strip("m")) * 60
    else:
        sleep_seconds = int(args.sleep)
    print("Going to sleep  seconds")

    final_run_list.insert(0, f"sleep {sleep_seconds}")
    return final_run_list


def run_cmds(run_list, cwd, args):
    run_async(run_list, parallel_num=args.parallel_num, cwd=cwd)


def do_everything(
    default_parameters,
    configs,
    extra_folder_keys,
    appendix_keys,
    main_script,
    test_parameters,
    wandb_project_name,
    script_file,
    args=None,
):
    """Function that will execute experiments in parallel or sequentially
    Args:
    default_parameters    : dict of default parameters
    configs               : list of (dict, path), each dict is a config, path tunnels output_dir (use None by default)"
    extra_folder_keys     : list of str, keys to use for the folder name
    appendix_keys         : list of str, keys to use for the appendix name
    main_script           : str, path to the main script
    test_parameters       : dict of parameters to use for testing
    wandb_project_name    : str, name of the wandb project
    script_file           : str, path to the script file
    """

    if args is None:
        args = get_script_args()

    work_dir = get_cwd(args, script_file)
    # configs = set_configs()

    folder_keys = [
        os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(script_file)))
        ),
        os.path.basename(os.path.dirname(os.path.dirname(script_file))),
        os.path.basename(os.path.dirname(script_file)),
        os.path.basename(script_file).split(".")[0],
    ]
    # extra_folder_keys, appendix_keys = get_name_keys()
    folder_keys += extra_folder_keys

    # if args.kill_concurrent_folders:
    #     parent_dir = os.path.dirname(args.output_dir)
    #     concurrent_folders = os.listdir(parent_dir)
    #     for folder in concurrent_folders:
    #         dir_to_remove = os.path.join(parent_dir, folder)
    #         # Dirname has to be specific format with time
    #         # For safety reasons
    #
    #         splits = folder.split('-')
    #         if len(splits) > 4 and 'T' in splits[-3]:
    #             shutil.rmtree(dir_to_remove)
    #     else:
    #         print(f"WARNING: Improper name, will not remove {dir_to_remove}")

    run_list = configs2cmds(
        configs,
        default_parameters,
        main_script,
        args,
        folder_keys,
        appedix_keys=appendix_keys,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
        work_dir=work_dir,
    )
    # print(run_list)
    run_cmds(run_list, work_dir, args)
