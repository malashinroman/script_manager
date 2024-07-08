# script_manager
Boilerplate code for managing scripts and logs with weights and biases

Consider side project.
1. Add script_manager as submodule (script_manager has to appear in the root of the project)

         git submodule add https://github.com/malashinroman/script_manager
2. Create ./scripts and ./tran_results folders in the project's root and copy script_manager/year/month/day/launch_script.py there. This is script to substitute bash scripts for providing parameters
3. Find where the project implement ArgumentParser and substitute parse_args() with smart_parse_args().
```python
    # in the master project's main script
    from script_manager.func.add_needed_args import smart_parse_args

    # I assume project uses argparse by default
    # replace parsing of the parameters
    # args = parser.parse_args()
    args = smart_parse_args(parser)

```
4. Find where the project saves data and make sure that from now it will save it to args.output_dir.
5. (optionaly) use write_wandb_scalar or write_wandb_dict to log data from the project.

Now you have control over the side project from script.
Results will be saved under <project_root>/train_results/script_date.


## How to use script_manager when it is already attached to project
1. Clone the project

   git clone [](https://github.com/malashinroman/)https://github.com/malashinroman/project-name
   
2.
  ```
  cd project-name
  git submodule init
  git submodule update
  ```
4. In the project's root:
   - modify `local_config.py` where local host information should be stored
   - create empy `train_results` folder
local_config.py example:
IMAGENET_PATH = "/media/Data2/datasets/ImageNet"
WANDB_LOGIN="newton"

5. Now you can try to run some script from `scripts` folder

`python scripts/2023/12/01/great_cool_script_for_training_agi_model.py`


## Scenarios

### Sequence of Scripts to be Launched in Parallel or Consecutively

Organize a master script that will prepare configurations:
```python
from script_manager.scenarios.parallel_sequence import do_sequence, parse_args
```

then get configs you want to run and finnaly launch everythin with do_sequece function

```python
    do_sequence(
        args,
        configs,
        default_parameters,
        MAIN_SCRIPT,
        __file__,
        test_parameters,
        WANDB_PROJECT_NAME,
    )
```
Put this master script inside the project scripts/y/m/d, where scripts/ is nearby script_manager. 

In shell exectue the master_script.py.

example1:
```bash
python scripts/24/04/23/master_script.py -n -e -g 1 2 3 --processes 3 --parallel_num 3 
```
- -n will remove test parameters
- -e will add logging to wandb
- -g will specifiy which gpus will be used (achieved by adding CUDA_VISIBLE_DEVICES under the hood)
- --parallel_num and --processes specify number of processes that will evoked and executed in parallel, 3 processes will wait for new configs until the end of everything 



## important parameters (script)

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


