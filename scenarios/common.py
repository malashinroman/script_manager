def create_launch_commands(main_script, gpus2use, configs_number):
    launch_commands = []
    for index in range(configs_number):
        if isinstance(gpus2use, list):
            gpu = gpus2use[index % len(gpus2use)]
            launch_commands.append(f"CUDA_VISIBLE_DEVICES={gpu} python {main_script}")
        else:
            # isinstance(gpus2use, int):
            launch_commands.append(main_script)
    return launch_commands
