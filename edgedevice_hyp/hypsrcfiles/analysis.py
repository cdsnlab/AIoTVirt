# simple program to cal total bw usage during runtime

import subprocess
import sys
import os
import fnmatch

ind1 = ['p_tsdr_sk44_'] 
#ind1 = ['cam1_2355_20'] # skip 60 for now

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
            if fnmatch.fnmatch(log_file, 'p_tsdr_sk44_*'): # found anchor logfile, open, read, save all coulumns to dictionary 
                f = open(log_file,'r')
                lines = f.readlines()
                temp = []
                for line in lines:
                    items = line.split()
                    if len(items) ==8: # if this is false, corrupted file.
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
                if(lbl[i][6] == str(1)):
                    dt += float(lbl[i][5])
                    det += float(lbl[i][4])
                    countone +=1
                    est += float(lbl[i][3])
            print("dq: ", str(dt / countone))
            print("det: ", str(det / countone))
            print("est: ", str(est / countone))

