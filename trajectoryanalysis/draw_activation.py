import matplotlib.pyplot as plt
import os
import pandas as pd
path_gt = "/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/gt_activation.xlsx"
path_bf ="/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/bf_activation.xlsx"
path_prev ="/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/prev_activation.xlsx"
path_prop ="/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/prop_activation.xlsx"

shname = "test-activation_number"
data_gt = pd.read_excel(path_gt, sheet_name=shname)
data_bf = pd.read_excel(path_bf, sheet_name=shname)
data_prev = pd.read_excel(path_prev, sheet_name=shname)
data_prop = pd.read_excel(path_prop, sheet_name=shname)
cnt=0
for col in data_prop.columns:
    gt_list=[]
    bf_list=[]
    prev_list=[]
    prop_list = []
    
    if col == "dummy":
        continue
    #print(col)
    gt_list = data_gt[col].tolist()
    bf_list = data_bf[col].tolist()
    prev_list = data_prev[col].tolist()
    prop_list = data_prop[col].tolist()
    for i in prop_list[:]:
        if str(i) == "-1":
            gt_list.remove(i)
            bf_list.remove(i)
            prev_list.remove(i)
            prop_list.remove(i)
    x = [*range(0, len(bf_list), 1)]
    plt.rcParams["figure.figsize"] = (10,4)
    plt.rcParams["axes.grid"] = True
    plt.xlabel("frame number")
    plt.ylabel("number of activeated cameras ")
    plt.yticks((0, 5, 10))
    plt.ylim(top=11)
    plt.ylim(bottom=0)
    plt.plot(x, gt_list, label="GT")
    plt.plot(x, bf_list, label="BF")
    plt.plot(x, prev_list, label = "PREV")
    plt.plot(x, prop_list, label = "PROP")
    plt.legend(loc='upper right')
    plt.savefig("/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/graph/"+col+".png")
    plt.clf()
    if cnt == 10:
        break
    cnt +=1

