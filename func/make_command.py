# import subprocess

import os
import datetime
from typing import Dict


base_params = None

def dict_to_string(dictionary):
    string = ''
    for (k, v) in dictionary.items():
        if k == 'load_checkpoint_DM':
            v = os.path.basename(os.path.dirname(v))
        if type(k) is not str:
            k = repr(k)
        if type(v) is not str:
            v = repr(v)
        if k != 'tag':
            string += '{}_{}_'.format(k, v).replace('.', '_')
        else:
            string += '{}_'.format(v).replace('.', '_')
    return string


def path_from_keylist(dict, keys):
    path = 'train_results'
    for k in keys:
        if k in dict:
            v = dict[k]
        else:
            v = k
        if type(v) is not str:
            v = repr(v)
        if type(k) is not str:
            k = repr(k)
        if k == v:
            folder = '{}'.format(v).replace('.', '_')
        else:
            folder = '{}_{}'.format(k, v).replace('.', '_')
        path = os.path.join(path, folder)
    return path

def make_command2(
        param_dict: Dict[str, str],
        script_name='main.py',
        folder_keys=[],
        appendix_keys=[]):
    '''
    Create text represenation of the command to be launhced with all the necessary non-default parameters
    By default adds '--output' argument with the values according
    to the state of global appendix_keys, folder_keys, which are set externally

    :param param_dict: dictionary with scripts argument names and values
    :param script_name: name of the script to be launched
    :return: tuple - (i) text representation of the command, (ii) path to the output folder
    '''

    # global folder_keys
    # global appendix_keys

    appendix_dict = {k: param_dict[k] for k in appendix_keys}
    appendix = dict_to_string(appendix_dict)
    time = datetime.datetime.now().isoformat(sep='T').replace(':', '-').replace('.', '_')
    path = path_from_keylist(param_dict, folder_keys)

    output = os.path.join(path, appendix + time)
    if not os.path.exists(output):
        print(output)
        os.makedirs(output)

    cmd = f"python {script_name} --output_dir={output} "

    for key, val in param_dict.items():
        if val != "parameter_without_value":
            cmd += f"--{key}={val} "
        else:
            cmd += f"--{key} "

    cmd = cmd[:-1]
    return cmd, output
