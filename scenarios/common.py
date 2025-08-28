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


def str2list(v: str) -> list[int]:
    # Accepts "0,1,2,3" or "[0,1,2,3]"
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]  # remove brackets
    return [int(x.strip()) for x in v.split(",") if x.strip()]
