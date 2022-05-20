import os
import sys
from copy import deepcopy
from pdb import run

from script_manager.make_command import make_command2

sys.path.append(".")
from local_config import DOMAIN_NET_PATH

sys.path.append(".")
import argparse
from pathlib import Path

from script_manager.run_series_of_experiments import run_async


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(',')]


def str2list(v):
    return [str(x.strip()) for x in v.strip()[1:-1].split(',')]


parser = argparse.ArgumentParser(description="TargetRecogniton")
parser.add_argument("--test", action="store_true")
parser.add_argument("--sleep", type=int, default=0)
parser.add_argument("--enable_wandb", action="store_true")
parser.add_argument("--parallel_num", type=int, default=1)
parser.add_argument("--configs2run", type=str2intlist, default=None)
parser.add_argument("--cwd", type=str, default=None)

args = parser.parse_args()


def get_main_script():
    main_script = os.path.join("tools", "train.py")
    return main_script

def get_wandb_project_name():
    return "debug_sparse_dassl"

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
    'epochs': 1,
    'model': 'DTRAM19',
    'ce_action_predictor_coeff': 1,
    'entropy_unc_coeff': 0.01,
    'entropy_var_coeff': 50,
    'forbid_short_episodes': True,
    'freeze_all_but_stop_net': 0,
    'gama': 1,
    'intermediate_supervision': 1,
    'kl_action_perdictor_coeff': 0,
    'learnable_std': False,
    'long_shot_baseline': 1,
    'loss_function': 'SPARSE_ENSEMBLE',
    'nll_reward': False,
    'num_glimpses': 2,
    'reinforce_act_coeff': 0.01,
    'reinforce_loc_coeff': 0.,
    'reinforce_stop_coeff': 0.,
    'rl_algorithm': 'dt_ram2',
    'semi_environment': 'cifar_env_responder',
    'std': 0.05,
    'std_entropy_coeff': 0.0,
    'stop_exploration_reg': 0,
    'supervised_loss_coeff': 1.,
    'train_patience': 200,
    'trainer': 'Trainer_full_model',
    'use_lstm': False,
    'use_mask_state': 0,
    'use_gpu': 1,
    'cifar_classifier_indexes': '[0,1,2]',
    'detach_classifier_features': 1,
    'n_classes': 126,
    'dataset-config-file': 'configs/datasets/da/mini_domainnet.yaml',
    "root": f"{DOMAIN_NET_PATH}",
    "source-domains": "[clipart,painting,real]",
    "tag": "default_tag",
    "target-domains": "[sketch]",
    'config-file': 'configs/trainers/da/sparse_dael/mini_domainnet.yaml',
}


def set_configs():
    configs = []
    config1 = \
        {
            'trainer': 'SPARSE_DAEL',
            'detach_classifier_features': 1,
            'tag': 'trest',
            'model-dir': 'checkpoints/learn_dael_2022-05-18T13-42-05_915643',
            'modules2load': "[E,F]",
        }

    configs.append([config1, None])
    config2 = \
        {
            'trainer': 'SPARSE_DAEL',
            'detach_classifier_features': 0,
            'tag': 'trest',
            'modules2load': '[E,F,AM]',
        }
    configs.append([config2, "model-dir"])

    return configs


def get_test_update_dict():
    d = {'config-file': 'configs/trainers/da/sparse_dael/mini_domainnet_test.yaml'}
    return d


def get_cwd():
    if args.cwd is None:
        cwd = str(path.parent.parent.parent.parent.parent.absolute())
    else:
        cwd = args.cwd
    print({'cwd': cwd})
    return cwd


def configs2cmds(full_configs, default_parameters, main_script, args):
    configs = [f[0] for f in full_configs]
    uof = [f[1] for f in full_configs]
    assert (len(uof) == len(configs))

    folder_keys, appedix_keys = get_name_keys()
    run_list = []
    for data_configuration, output_forward_key in zip(configs, uof):
        configuration_dict = deepcopy(default_parameters)
        configuration_dict.update(data_configuration)
        if output_forward_key is not None:
            if len(output_forward_key) > 0:
                assert(pref_step_output is not None)
                configuration_dict[output_forward_key] = pref_step_output
        if args.test:
            test_parameters = get_test_update_dict()
            configuration_dict.update(test_parameters)
        if args.enable_wandb:
            configuration_dict['wandb_project_name'] = get_wandb_project_name()
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
    path = Path(__file__)
    cwd = get_cwd()
    configs = set_configs()

    run_list = configs2cmds(configs, default_parameters, get_main_script(), args)
    run_cmds(run_list, cwd, args)
