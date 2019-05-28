# simple program to see the percentage of target in two videos.
# check if at least one of the video has the person inside!

import subprocess
import sys
import os
import fnmatch

ind1 = ['cam1_0000', 'cam2_0000', 'cam1_0001', 'cam2_0001','cam1_2349', 'cam2_2349','cam1_2352', 'cam2_2352','cam1_2358', 'cam2_2358',] 
#ind1 = ['cam1_2355_20'] # skip 60 for now

if __name__ == '__main__': 
    for i in ind1: #find files with index names. 
        proc = subprocess.Popen('ls *_{}*.txt'.format(i), shell=True, stdout=subprocess.PIPE)
        ls_result_b = proc.stdout.read()
        ls_result_d = ls_result_b.decode('ascii')
        log_files = ls_result_d.split()
        lbl = {}
        target =0.0
        cnt =0.0
        for log_file in log_files: # need to print last line.
            lbl = {}
            if fnmatch.fnmatch(log_file, 'e1g*'): # found anchor logfile, open, read, save all coulumns to dictionary 
                print("groundtruth: ", log_file)
                f = open(log_file,'r')
                lines = f.readlines()
                temp = []
                for line in lines:
                    items = line.split()
                    if len(items) ==4: # if this is false, corrupted file.
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
            else:
                continue
            for i in (lbl):
#                print (lbl[i][2])
                if lbl[i][2] == "1":
                    target += 1
                cnt +=1
            print (log_file + "\t" + str(target / cnt) + "\t" + str(cnt)+"\t"+str(target))
