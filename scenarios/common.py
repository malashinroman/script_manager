def create_launch_commands(main_script, gpus2use, configs_number):
    launch_commands = []
    script = main_script
    for index in range(configs_number):
        if isinstance(main_script, list):
            script = main_script[index]
        if isinstance(gpus2use, list):
            gpu = gpus2use[index % len(gpus2use)]
            launch_commands.append(f"CUDA_VISIBLE_DEVICES={gpu} python {script}")
        else:
            launch_commands.append(script)
    return launch_commands
