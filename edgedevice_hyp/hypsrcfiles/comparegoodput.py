# simple program to cal total bw usage during runtime

import subprocess
import sys
import os
import fnmatch

ind1 = ['cam1_2046_60', 'cam2_2046_60' ] 
#ind1 = ['cam1_2355_20'] # skip 60 for now

if __name__ == '__main__': 
    for i in ind1: #find files with index names. 
        proc = subprocess.Popen('ls *_{}*.txt'.format(i), shell=True, stdout=subprocess.PIPE)
        ls_result_b = proc.stdout.read()
        ls_result_d = ls_result_b.decode('ascii')
        log_files = ls_result_d.split()
        lbl = {}
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
#            print(lbl)
#                print("length of lbl: ", len(lbl))
                break

        for log_file in log_files: # compare with other files
            TP = 0.0
            TN = 0.0
            FP = 0.0
            FN = 0.0
            templbl = {}
            if fnmatch.fnmatch(log_file, 'e1g*'):
                continue
            else:
                print( "comparison file: ", log_file)
                f = open(log_file,'r')
                lines = f.readlines()
                temp = []
                for line in lines:
                    items = line.split()
                    if len(items) ==4: # if this is false, corrupted file.
                        for item in items:
                            temp.append(item)
                        templbl[items[0]] = temp
#                        print(items[0], temp)
                        temp=[]
                    else:
                        print("corrupted file, exiting\n")
                        sys.exit(1)
                f.close()
#                print("length of templbl: ", len(templbl))
#                print("contents of templbl: ", str(templbl))
            for cake in templbl:
                if str(cake) == str(lbl[cake][0]):
                    if(templbl[cake][2] == str(1) and  lbl[cake][2]== str(1)):
                        TP +=1
                    elif(templbl[cake][2] == str(0) and lbl[cake][2] == str(0)):
                        TN +=1
                    elif(templbl[cake][2] == str(1) and lbl[cake][2] ==str(0)):
                        FP +=1
                    elif(templbl[cake][2] == str(0) and lbl[cake][2] ==str(1)):
                        FN +=1
            TN += len(lbl) - len(templbl)
            print "TP, TN, FP, FN: ", TP, TN, FP, FN
            print "accuracy: ", (TP + TN) / (TP + TN +FP +FN)
            print "precision: ", TP / (TP+FP)
            print "recall: ", TP / (TP+FN)
                    
            templbl = {}
