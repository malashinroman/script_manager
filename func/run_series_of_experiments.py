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


def get_attach_gpu_dict(run_list=[]):
    # convert run_list to dict with CUDA_VISIBLE_DEVICES inex as a key
    run_dict = {}
    unattached_list = []
    for cmd in run_list:
        _, my_env = get_command_params_environs(cmd)
        if "CUDA_VISIBLE_DEVICES" in my_env:
            gpu = int(my_env["CUDA_VISIBLE_DEVICES"])
            if gpu not in run_dict:
                run_dict[gpu] = []
            run_dict[gpu].append(cmd)
        else:
            unattached_list.append(cmd)
    return run_dict, unattached_list


def run_simple_parallel(run_list, parallel_num=2, cwd="."):
    progresses = []
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


def run_process_per_gpu(gpu_cmd_dict, parallel_num=2, cwd="."):
    gpu_num = len(gpu_cmd_dict)

    progresses_dict = {g: [] for g in range(gpu_num)}
    if isinstance(parallel_num, int):
        proc_per_gpu = [parallel_num // gpu_num] * gpu_num

    for i in range(gpu_num):
        for j in range(proc_per_gpu[i]):
            cmd = gpu_cmd_dict[i][j]
            command_params, my_env = get_command_params_environs(cmd)
            progresses_dict[i].append(
                subprocess.Popen(command_params, cwd=cwd, env=my_env)
            )
            gpu_cmd_dict[i] = gpu_cmd_dict[i][1:]

    for gpu_ind, gpu_progresses in progresses_dict.items():
        for progress in gpu_progresses:
            print(progress)

    while sum([len(v) for _, v in gpu_cmd_dict.items()]) > 0:
        for gpu_ind, gpu_progresses in progresses_dict.items():
            for i in range(len(gpu_progresses)):
                print(
                    f"gpu {gpu_ind} process {i} running: {gpu_progresses[i].poll() is None}"
                )
                if not gpu_progresses[i].poll() is None:
                    if len(gpu_cmd_dict[gpu_ind]) > 0:
                        cmd = gpu_cmd_dict[gpu_ind][0]
                        gpu_cmd_dict[gpu_ind] = gpu_cmd_dict[gpu_ind][1:]
                        command_params, my_env = get_command_params_environs(cmd)
                        gpu_progresses[i] = subprocess.Popen(
                            command_params, cwd=cwd, env=my_env
                        )
                        print(
                            f"gpu {gpu_ind} process {i} running: {gpu_progresses[i].poll() is None}"
                        )
        time.sleep(10)

    print("All Threads are queued, let's see when they finish!")

    for gpu_ind, gpu_progresses in progresses_dict.items():
        for p in gpu_progresses:
            p.wait()
    print("Everything finished!")


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

    assert len(run_list) >= parallel_num, "run_list should be longer than parallel_num"

    gpu_attached, unattached_list = get_attach_gpu_dict(run_list)

    if len(unattached_list) > 0:
        run_simple_parallel(run_list, parallel_num, cwd)
    else:
        run_process_per_gpu(gpu_attached, parallel_num, cwd)
