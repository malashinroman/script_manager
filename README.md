# script_manager
Boilerplate code for managing scripts and logs with weights and biases

Consider side project.
1. Add script_manager as submodule (script_manager has to appear in the root of the project).
2. Create ./scripts folder in the project root and copy script_manager/year/month/day/launch_script.py there. This is script to substitute bash scripts for providing parameters
3. Find where the project implement ArgumentParser and substitute parse_args() with smart_parse_args().
4. Find where the project saves data and make sure that from now it will save it to args.output_dir().
5. (optionaly) use write_wandb_scalar or write_wandb_dict to log data from the project.

Now you have control over the side project from script.
Results will be saved under <project_root>/train_results/script_date.


