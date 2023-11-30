# script_manager
Boilerplate code for managing scripts and logs with weights and biases

Consider side project.
1. Add script_manager as submodule (script_manager has to appear in the root of the project)

         git submodule add https://github.com/malashinroman/script_manager
2. Create ./scripts folder in the project's root and copy script_manager/year/month/day/launch_script.py there. This is script to substitute bash scripts for providing parameters
3. Find where the project implement ArgumentParser and substitute parse_args() with smart_parse_args().
4. Find where the project saves data and make sure that from now it will save it to args.output_dir.
5. (optionaly) use write_wandb_scalar or write_wandb_dict to log data from the project.

Now you have control over the side project from script.
Results will be saved under <project_root>/train_results/script_date.


# How to use script_manager when it is already attached to project
1. Clone the project

   `git clone [<hh](https://github.com/malashinroman/)https://github.com/malashinroman/<project-name>
   
2.
   `cd <project-name>

    git submodule init
  
    git submodule update`

3. In the project's root:
   - modify `local_config.py` where local host information should be stored
   - create empy `train_results` folder
local_config.py example:
IMAGENET_PATH = "/media/Data2/datasets/ImageNet"
WANDB_LOGIN="newton"

4. Now you can try to run some script from `scripts` folder

`python scripts/2023/12/01/great_cool_script_for_training_agi_model.py`

# important parameters to scripts

```
   parser.add_argument(
        "-n",
        "--not_test",
        action="store_true",
        help="if set to True, than test_parameters will be ignored",
    )
    parser.add_argument(
        "--sleep",
        type=str,
        default="0",
        help="sleep time before starting the script (10s - 10 seconds, 10m -10 minutes, 10h - 10 hours)",
    )
    parser.add_argument(
        "-e",
        "--enable_wandb",
        action="store_true",
        help="if set to True, than wandb will be enabled",
    )
    parser.add_argument(
        "--parallel_num",
        type=int,
        default=1,
        help="specifies number of processes in parallel run, if set to 2, than the script will be run in parallel with 2 processes",
    )
    parser.add_argument("-c", "--configs2run", type=int, nargs="*",
                        default=None, help="list of indexes of scripts to run (for debug)")
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
```


