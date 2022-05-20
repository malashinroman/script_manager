import threading
import time
import random
import subprocess

def run_async(run_list=[], parallel_num=2, cwd='.'):
    if parallel_num == 1:
        for cmd in run_list:
            lol = cmd.split(" ")
            subprocess.call(cmd.split(" "), cwd=cwd)
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
