import os
import sys

sys.path.append(".")

from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import get_script_args

from local_config import IMAGENET_PATH

args = get_script_args()

# script to be used
main_script = os.path.join("script_manager", "main", "hello_imagenet.py")

# weights and biases project name
wandb_project_name = "hello_imagenet"

# keys
appendix_keys = ["tag"]
extra_folder_keys = []

# default parameteres
default_parameters = {"imagenet_path": IMAGENET_PATH}
test_parameters = {"number_of_iterations": 0}

# configs to be exectuted
configs = []
config1 = {"number_of_iterations": 1, "tag": "num_1_"}

configs.append([config1, None])
config2 = {"number_of_iterations": 10, "tag": "num_10_"}

configs.append([config2, "prev_result"])
# RUN everything
# !normally you don't have to change anything here
if __name__ == "__main__":
    do_everything(
        default_parameters=default_parameters,
        configs=configs,
        extra_folder_keys=extra_folder_keys,
        appendix_keys=appendix_keys,
        main_script=main_script,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
        script_file=__file__,
    )
