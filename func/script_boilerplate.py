import os
from copy import deepcopy
from pathlib import Path
from script_manager.func.make_command import make_command2
from script_manager.func.script_parse_args import get_script_args
from script_manager.func.run_series_of_experiments import run_async


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


# def get_name_keys():
#     # this parameters will be logged as appendix of the directory name
#     folder_keys = []
#
#     # this parameters will be logged as as subfolders in directory path
#     appendix_keys = ["tag"]
#
#     return folder_keys, appendix_keys


def get_cwd(args, file_path):
    path = Path(file_path)
    if args.cwd is None:
        cwd = str(path.parent.parent.parent.parent.parent.absolute())
    else:
        cwd = args.cwd
    print({"cwd": cwd})
    return cwd


def configs2cmds(
    full_configs,
    default_parameters,
    main_script,
    args,
    folder_keys,
    appedix_keys,
    test_parameters,
    wandb_project_name,
):
    configs = [f[0] for f in full_configs]
    uof = [f[1] for f in full_configs]
    assert len(uof) == len(configs)

    # folder_keys, appedix_keys = get_name_keys()
    run_list = []
    for data_configuration, output_forward_key in zip(configs, uof):
        configuration_dict = deepcopy(default_parameters)
        configuration_dict.update(data_configuration)
        if output_forward_key is not None:
            if len(output_forward_key) > 0:
                assert pref_step_output is not None
                configuration_dict[output_forward_key] = pref_step_output
        if not args.not_test:
            # test_parameters = get_test_update_dict(configuration_dict)
            # configuration_dict.update(test_parameters)
            update_with_test_dict(configuration_dict, test_dict=test_parameters)
            print("WARNING: test mode is activated")
        if args.enable_wandb:
            configuration_dict["wandb_project_name"] = wandb_project_name
        else:
            print("WARNING: no wandb logs are enabled")

        cmd0, pref_step_output = make_command2(
            configuration_dict, main_script, folder_keys, appedix_keys
        )
        # pref_step_output = output0
        run_list.append(cmd0)

    if args.configs2run is not None:
        final_run_list = [run_list[i] for i in args.configs2run]
    else:
        final_run_list = run_list

    final_run_list.insert(0, f"sleep {args.sleep}")
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
):

    args = get_script_args()

    work_dir = get_cwd(args, script_file)
    # configs = set_configs()

    folder_keys = [
        os.path.basename(os.path.dirname(os.path.dirname(script_file))),
        os.path.basename(os.path.dirname(script_file)),
        os.path.basename(script_file).split(".")[0],
    ]
    # extra_folder_keys, appendix_keys = get_name_keys()
    folder_keys += extra_folder_keys
    run_list = configs2cmds(
        configs,
        default_parameters,
        main_script,
        args,
        folder_keys,
        appedix_keys=appendix_keys,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
    )
    run_cmds(run_list, work_dir, args)
