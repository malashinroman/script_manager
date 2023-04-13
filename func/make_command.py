import datetime
import os
from typing import Dict

base_params = None


def dict_to_string(dictionary: dict) -> str:
    string = ""
    for (k, v) in dictionary.items():
        if k == "load_checkpoint_DM":
            v = os.path.basename(os.path.dirname(v))
        if type(k) is not str:
            k = repr(k)
        if type(v) is not str:
            v = repr(v)
        if k != "tag":
            string += f"{k}_{v}_".replace(".", "_")
        else:
            string += f"{v}_".replace(".", "_")
    return string


def path_from_keylist(dictionary: dict, keys: list) -> str:
    path = "train_results"
    for k in keys:
        if k in dictionary:
            v = dictionary[k]
        else:
            v = k
        if type(v) is not str:
            v = repr(v)
        if type(k) is not str:
            k = repr(k)
        if k == v:
            folder = "{}".format(v).replace(".", "_")
        else:
            folder = "{}_{}".format(k, v).replace(".", "_")
        path = os.path.join(path, folder)
    return path


def make_command2(
    param_dict: Dict[str, str],
    script_name: str | list[str] = "",
    folder_keys: list[str] = [],
    appendix_keys: list[str] = [],
    work_dir: str = "",
):
    """
    Create text represenation of the command to be launhced with all the necessary non-default parameters
    By default adds '--output' argument with the values according
    to the state of global appendix_keys, folder_keys, which are set externally

    : param param_dict  : dictionary with scripts argument names and values
    : param script_name : name of the script to be launched
    : return            : tuple - (i) text representation of the command, (ii) path to the output folder
    """

    # global folder_keys
    # global appendix_keys

    appendix_dict = {k: param_dict[k] for k in appendix_keys}
    appendix = dict_to_string(appendix_dict)
    time = (
        datetime.datetime.now().isoformat(sep="T").replace(":", "-").replace(".", "_")
    )
    path = path_from_keylist(param_dict, folder_keys)

    output = os.path.join(work_dir, path, appendix + time)
    if not os.path.exists(output):
        os.makedirs(output)

    cmd = ""
    if isinstance(script_name, str):
        if len(script_name.split(" ")[0]) > 0:
            fist_c = script_name.split(" ")
            if fist_c[0].endswith(".py"):
                cmd = f"python "
            elif fist_c == 'torchrun':
                pass
                # add ports to enable multiple instances for
                # multiple ports

    cmd += f"{script_name} "
    if not "output_dir" in param_dict:
        cmd += f"--output_dir={output} "
        print(output)
    else:
        print(param_dict["output_dir"])
        assert not os.path.exists(param_dict["output_dir"])

    if "__script_output_arg__" in param_dict:
        script_arg = param_dict["__script_output_arg__"]
        cmd += f"--{script_arg}={output} "
        del param_dict["__script_output_arg__"]

    for key, val in param_dict.items():
        if val != "parameter_without_value":
            cmd += f"--{key}={val} "
        else:
            cmd += f"--{key} "

    cmd = cmd[:-1]
    return cmd, output
