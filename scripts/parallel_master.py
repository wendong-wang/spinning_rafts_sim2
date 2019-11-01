#!/usr/bin/env python
"""
This is for running simulation files in parallel.
"""
import multiprocessing
import subprocess
# import sys
import os
from multiprocessing.pool import ThreadPool
import numpy as np
import platform


def process_files():
    pool = multiprocessing.pool.ThreadPool(6)  # Number of processors to be used
    if platform.node() == 'NOTESIT43' and platform.system() == 'Windows':
        project_dir = "D:\\simulationFolder\\spinning_rafts_sim2"
    elif platform.node() == 'NOTESIT71' and platform.system() == 'Linux':
        project_dir = r'/media/wwang/shared/spinning_rafts_simulation/spinning_rafts_sim2'
    else:
        project_dir = os.getcwd()

    if project_dir != os.getcwd():
        os.chdir(project_dir)

    data_dir = os.path.join(project_dir, 'data')
    script_dir = os.path.join(project_dir, "scripts")
    filename = 'simulation_combined.py'
    num_of_rafts = [6]  # num of rafts
    spin_speeds = np.arange(-10, -20, -1)  # spin speeds, negative means clockwise in the rh coordinate

    for arg1 in num_of_rafts:
        for arg2 in spin_speeds:
            script_file = os.path.join(script_dir, filename)
            # print(script_file)
            cmd = ["python", script_file, str(arg1), str(arg2)]

            #        p=subprocess.check_call(cmd ,stdout=subprocess.PIPE)
            #        print(p.communicate())
            #        print(script_file)
            # pool.apply_async(cmd) # supply command to system
            print(cmd)
            pool.apply_async(subprocess.check_call, (cmd,))  # supply command to system
    pool.close()
    pool.join()


if __name__ == '__main__':
    process_files()
