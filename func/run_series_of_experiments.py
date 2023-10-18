import os
import subprocess
import time

ENVIRONS_TO_GET = ["CUDA_VISIBLE_DEVICES"]


def get_command_params_environs(cmd: str):
    """Get command, params and environs from a string."""
    cmd_words = cmd.split(" ")
    command_params = []
    environs = {}
    my_env = os.environ.copy()
    for w in cmd_words:
        # Is this a good way to check if this is an environ?
        if any([e in w for e in ENVIRONS_TO_GET]):
            assert (
                "=" in w
            ), f"environs should be in the form of key=value, but {w} is not"
            environs[w.split("=")[0]] = w.split("=")[1]
            # alternative way to check if this is an environs
        else:
            command_params.append(w)

    for k, v in environs.items():
        my_env[k] = v

    return command_params, my_env


def run_async(run_list=[], parallel_num=2, cwd="."):
    """
    Run a list of commands in parallel.
    param run_list     : list of commands to be run
    param parallel_num : number of commands to be run in parallel
    param cwd          : current working directory
    return             :
    """
    if parallel_num == 1:
        for cmd in run_list:
            print(f"cmd:{cmd}")

            """launch the process"""
            command_params, my_env = get_command_params_environs(cmd)
            subprocess.call(command_params, cwd=cwd, env=my_env)
            # subprocess.Popen(cmd.split(" "),shell=True)
        return

    progresses = []
    assert len(run_list) >= parallel_num, "run_list should be longer than parallel_num"
    for cmd in run_list[:parallel_num]:
        command_params, my_env = get_command_params_environs(cmd)
        progresses.append(subprocess.Popen(command_params, cwd=cwd, env=my_env))

    run_list = run_list[parallel_num:]

    for i in range(len(progresses)):
        print(progresses[i])

    while len(run_list) > 0:
        for i in range(len(progresses)):
            print(f"process {i} running: {progresses[i].poll() is None}")
            if not progresses[i].poll() is None:
                cmd = run_list[0]
                run_list = run_list[1:]
                command_params, my_env = get_command_params_environs(cmd)
                progresses[i] = subprocess.Popen(command_params, cwd=cwd, env=my_env)
                print(f"process {i} running: {progresses[i].poll() is None}")
        time.sleep(10)

    print("All Threads are queued, let's see when they finish!")

    for p in progresses:
        p.wait()
    print("Everything finished!")
