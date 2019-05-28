# simple program to cal total bw usage during runtime

import subprocess
import sys
import os
import fnmatch
ind1 = ['0000', '0001', '2349', '2352', '2358']

#ind1 = ['cam1_2046','cam2_2046', 'cam1_2046','cam2_0001', 'cam1_2349','cam2_2349', 'cam1_2352','cam2_2352', 'cam1_2356', 'cam2_2356' ] 
#ind1 = ['cam1_2355_20'] # skip 60 for now

if __name__ == '__main__': 
    for i in ind1: #find files with index names. 
        proc = subprocess.Popen('ls *_{}*.txt'.format(i), shell=True, stdout=subprocess.PIPE)
        ls_result_b = proc.stdout.read()
        ls_result_d = ls_result_b.decode('ascii')
        log_files = ls_result_d.split()
        lbl = {}
        cutline = 1.0
        for log_file in log_files: # need to print last line.
            if fnmatch.fnmatch(log_file, '*gpu*'):
                continue
            lbl = {}
            print("currentfile: ", log_file)
            f = open(log_file,'r')
            lines = f.readlines()
            temp = []
            add = 0.0
            for line in lines:
                items = line.split()
                if len(items) ==4: # if this is false, corrupted file.
                    for item in items:
                        temp.append(item)
#                     print temp
                    lbl[items[0]] = temp
#                        lbl.append(temp)
                    temp=[]
                else:
                    print("corrupted file, exiting\n")
                    sys.exit(1)
            f.close()
#                print "len lbl, ", len(lbl)
            for cake in lbl.keys():
#            print float(lbl[cake][1])
                if (float(lbl[cake][1]) <= cutline):
                    add+=1
        
            print add / len(lbl)
