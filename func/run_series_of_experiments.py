import subprocess
import time


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
            subprocess.call(cmd.split(" "), cwd=cwd)
            # subprocess.Popen(cmd.split(" "),shell=True)
        return

    progresses = []
    assert len(run_list) >= parallel_num
    for cmd in run_list[:parallel_num]:
        progresses.append(subprocess.Popen(cmd.split(" ")))

    run_list = run_list[parallel_num:]

    for i in range(len(progresses)):
        print(progresses[i])

    while len(run_list) > 0:
        for i in range(len(progresses)):
            print(f"process {i} running: {progresses[i].poll() is None}")
            if not progresses[i].poll() is None:
                cmd = run_list[0]
                run_list = run_list[1:]
                progresses[i] = subprocess.Popen(cmd.split(" "))
                print(f"process {i} running: {progresses[i].poll() is None}")
        time.sleep(10)

    print("All Threads are queued, let's see when they finish!")

    for p in progresses:
        p.wait()
    print("Everything finished!")
