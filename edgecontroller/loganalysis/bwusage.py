# simple program to cal total bw usage during runtime

import subprocess
import sys
import os

if __name__ == '__main__':
    cam = sys.argv[1]
    proc = subprocess.Popen('ls *_{}*.txt'.format(cam), shell=True, stdout=subprocess.PIPE)
    ls_result_b = proc.stdout.read()
    ls_result_d = ls_result_b.decode('ascii')
    log_files = ls_result_d.split()

    for log_file in log_files: # need to print last line.
        with open(log_file, "rb") as f:
            first = f.readline()
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
               f.seek(-2, os.SEEK_CUR)
            last = f.readline()
            print(log_file +"\t"+str(last))
