import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


def createsheetname(variable):
    transitionmodels = ["conv_lstm"]
    timemodels = ["ResNet", "dt", "rf"]
    vls = [15, 30]
    preprocessingmethods = ["last", "ed"]
    shnames = []
    if variable == "timemodels":
        for i in timemodels:
            shnames.append(transitionmodels[0]+"_"+i+"_"+str(vls[0])+"_"+preprocessingmethods[0])
    elif variable =="transitionmodels":
        for i in transitionmodels:
            shnames.append(i+"_"+timemodels[0]+"_"+str(vls[0])+"_"+preprocessingmethods[0])
    elif variable =="vls":
        for i in vls:
            shnames.append(transitionmodels[0]+"_"+timemodels[0]+"_"+str(i)+"_"+preprocessingmethods[0])
    elif variable =="preprocessingmethods":
        for i in preprocessingmethods:
            shnames.append(transitionmodels[0]+"_"+timemodels[0]+"_"+str(vls[0])+"_"+i)
    
    return shnames
        
result_path = "/home/spencer1/AIoTVirt/trajectoryanalysis/results/scenario_newsimdata_prop_variation.xlsx"
data={}
accvalue={}
prevalue={}
target = "preprocessingmethods"
#! what do i want for x axis?
#! 1) X: varying ML models (time_est) Y1: precision, Y2: Acc 
#! 2) X: varying ML models (dir_est) Y1: precision, Y2: Acc
#! 3) X: varying VL sizes (15, 30) Y1: precision, Y2: Acc
#! 4) X: varying PP methods (last, ed, sw-o, irw) Y1: precision, Y2: Acc
#! - fix other variables
#* load files, open sheets with the variable u want to change.
for i, sheetname in enumerate(createsheetname(target)):
    data[i] = pd.read_excel(result_path, sheet_name=sheetname)

    for index, col in enumerate(data[i].columns):
        if col == "accuracy":
            #* average all from that column
            print(data[i][col].mean())
            accvalue[sheetname] =data[i][col].mean()
        
        if col == "precision":
            #* average all from that column
            print(data[i][col].mean())
            prevalue[sheetname] =data[i][col].mean()

t = 2
barWidth = 0.8
n=1
d = len(accvalue)

x1_values = [t*element + barWidth*n for element in range(d)]
plt.bar(x1_values, list(accvalue.values()), color='r', label="accuracy")

x2_values = [t*element + barWidth*(n+1) for element in range(d)]
plt.bar(x2_values, list(prevalue.values()), color='b', label="precision")

plt.grid(True)
plt.title("accuracy & precision with varying {}".format(target))
# plt.bar(list(accvalue.keys()), list(accvalue.values()), barWidth)
# plt.bar(list(prevalue.keys()), list(prevalue.values()), barWidth)
plt.savefig("/home/spencer1/AIoTVirt/trajectoryanalysis/final_graphs/"+target+".png")
plt.clf()
