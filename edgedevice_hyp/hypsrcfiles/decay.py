# simple program to cal decay from all log files.

import subprocess
import sys


if __name__ == '__main__':
    cam = sys.argv[1]
    proc = subprocess.Popen('ls *_{}*.txt'.format(cam), shell=True, stdout=subprocess.PIPE)
    ls_result_b = proc.stdout.read()
    ls_result_d = ls_result_b.decode('ascii')
    log_files = ls_result_d.split()
    cams_decay_time = []

    for log_file in log_files:
        detect_count = 0
        decay_time_sum = 0.0

        f = open(log_file, 'r')
        lines = f.readlines()
        for line in lines:
            items = line.split()
            if len(items) == 4:
                if int(items[2]) == 1:
                    detect_count += 1
                    decay_time_sum += float(items[1])
        f.close()

        if detect_count > 0:
            print(log_file+"\t"+ str(decay_time_sum/detect_count))
        else:
            print(' - log invalid (no data)')
