# simple program to cal total bw usage during runtime

import subprocess
import sys
import os
import fnmatch

ind1 = ['cam1_0000','cam2_0000', 'cam1_0001','cam2_0001', 'cam1_2349','cam2_2349', 'cam1_2352','cam2_2352', 'cam1_2356', 'cam2_2356' ] 
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
#                        print temp
                        lbl[items[0]] = temp
#                        lbl.append(temp)
                        temp=[]
                    else:
                        print("corrupted file, exiting\n")
                        sys.exit(1)
                f.close()
#                print "len lbl, ", len(lbl)
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
#                        print temp
                        temp=[]
                    else:
                        print("corrupted file, exiting\n")
                        sys.exit(1)
                f.close()
#                print("length of templbl: ", len(templbl))
#                print("contents of templbl: ", str(templbl))
#            print "len templbl, ", len(templbl)
            for i in range(len(lbl)):
                if str(i) in templbl:
#                    print i, "exist"
                    pass
                else:
#                    print i, "inserting"
                    templbl[str(i)] = [str(i), '0', '0', '0']
#            print "len adj templbl, ", (templbl)
            for cake in lbl.keys():
#            for cake in range(len(templbl)):
                if cake == str(lbl[cake][0]):
                    if(templbl[cake][2] == str(1) and  lbl[cake][2]== str(1)):
                        TP +=1
                    elif(templbl[cake][2] == str(0) and lbl[cake][2] == str(0)):
                        TN +=1
                    elif(templbl[cake][2] == str(1) and lbl[cake][2] ==str(0)):
                        FP +=1
                    elif(templbl[cake][2] == str(0) and lbl[cake][2] ==str(1)):
                        FN +=1
            TN += len(lbl) - len(templbl)
            acc = (TP + TN) / (TP + TN + FP +FN)
            pre = TP / (TP + FP)
            rec = TP / (TP + FN)
#            f1s = 2 * (pre * rec) / (pre + rec)
            print "TP, TN, FP, FN: ", TP, TN, FP, FN
            print "accuracy: ", (TP + TN) / (TP + TN +FP +FN)
            print "precision: ", TP / (TP+FP)
            print "recall: ", TP / (TP+FN)
#            print "f1score: ", f1s
                    
            templbl = {}
