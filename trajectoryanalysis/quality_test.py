'''
this evaluates if the camera prediction is wrong or prediction time is wrong.
'''

import os
import pandas as pd
import re

path1 ="/home/spencer1/AIoTVirt/trajectoryanalysis/evaluate_prop_dir_or_time.xlsx"
#sheet1="even"
#data1 = pd.read_excel(path1, sheet_name=sheet1)
sheet2="odd-fixed_REC"
data1 = pd.read_excel(path1, sheet_name=sheet2)



def convert(row):
    row = row.strip('[]')
    row = list(re.split(', ',row))
    return row

#! we are long for ONLY 4 cases. Count transition of these following cases.
#!      gt   prop
#! 1)   1    -1 
#! 2)  -1     1
#! 3)   1     1
#! 4)  -1    -1 

sumwrongtime=0
sumwrontdir=0
end_off=0
cnt_wrong_time={i: 0 for i in range(int(len(data1.columns)/2))}
cnt_wrong_dir={i: 0 for i in range(int(len(data1.columns)/2))}
for i in range (int(len(data1.columns)/2)): 
#for i in range(40): #* iterate files.
    rightset=[]
    lates=0
    wrong_esitmation=0
    beginning=True
    rowsize = len(data1['gt_{}'.format(i)])-data1['gt_{}'.format(i)].isna().sum()
    #print(rowsize)
    diffstreak=0
    for index, row in data1.iterrows(): #* read frame by frame.
        if(str(row['gt_{}'.format(i)]))=="nan":
            print("endfile diffstreak {}, file {}".format(lates, i))
            #cnt_wrong_time[i] += diffstreak
            if lates > 5:
                end_off +=1
            break
        gt_values = convert(row['gt_{}'.format(i)])
        prop_value = (row['prop_{}'.format(i)].strip('[]'))
        #print("row number {}, gt {}, prop {}".format(index, gt_values, prop_value))
        if beginning==True:
            if int(prop_value) != -1: #* we've found smth
                beginning=False
                continue
                
        if beginning==False: #* beginning is FALSE
            print(prop_value, gt_values)
            if prop_value not in gt_values :       
                if diffstreak > 2:
                    rightset = gt_values
                diffstreak+=1
                if int(prop_value) != -1:
                    print("wrong estimation {}".format(wrong_esitmation))
                    cnt_wrong_dir[i]+=1
                    
            else: #* they are same. 
                if len(rightset)!=0:
                    if int(prop_value) != -1:
                        print("diffstreak {}".format(diffstreak))
                        cnt_wrong_time[i] += diffstreak
                        diffstreak=0
                        rightset=[]
           

for k,v in cnt_wrong_time.items():
    sumwrongtime += v
for k,v in cnt_wrong_dir.items():
    sumwrontdir += v
print("average difference in estimated time {}".format(sumwrongtime / int(len(data1.columns)/2)))
print("ending way off {} ".format(end_off / int(len(data1.columns)/2)))