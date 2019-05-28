# simple program to cal total bw usage during runtime

import subprocess
import sys
import os
import fnmatch

ind1 = ['e1-1time'] 

if __name__ == '__main__': 
    for i in ind1: #find files with index names. 
        proc = subprocess.Popen('ls {}*.txt'.format(i), shell=True, stdout=subprocess.PIPE)
        ls_result_b = proc.stdout.read()
        ls_result_d = ls_result_b.decode('ascii')
        log_files = ls_result_d.split()
        dt = 0.0
        det = 0.0
        est = 0.0
        countone = 0.0
        lbl = {}
        for log_file in log_files: # need to print last line.
            lbl = {}
            if fnmatch.fnmatch(log_file, 'e1-1time*'): # found anchor logfile, open, read, save all coulumns to dictionary 
                f = open(log_file,'r')
                lines = f.readlines()
                temp = []
                for line in lines:
                    items = line.split()
                    if len(items) == 4: # if this is false, corrupted file.
                        for item in items:
                            temp.append(item)
#                        print(temp)
                        lbl[items[0]] = temp
#                        lbl.append(temp)
                        temp=[]
                    else:
                        print("corrupted file, exiting\n")
                        sys.exit(1)
                f.close()
            print(len(lbl))

            for i in (lbl):
                countone +=1

                dt += float(lbl[i][3])
                det += float(lbl[i][2])
            print("dt: ", str(dt / countone))
            print("det: ", str(det / countone))

