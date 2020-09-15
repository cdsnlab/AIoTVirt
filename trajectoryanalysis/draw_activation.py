import matplotlib.pyplot as plt
import os
import pandas as pd
path_gt = "/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/gt_newsimdata_activation.xlsx"
path_bf ="/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/bf_newsimdata_activation.xlsx"
path_prev ="/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/prev_newsimdata_activation.xlsx"
path_prop ="/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/prop_newsimdata_activation.xlsx"

shname = "final"
prevshname = "fixed_wait_time"
propshname = "resnet_last0.4"

data_gt = pd.read_excel(path_gt, sheet_name=shname)
data_bf = pd.read_excel(path_bf, sheet_name=shname)
data_prev = pd.read_excel(path_prev, sheet_name=prevshname)
data_prop = pd.read_excel(path_prop, sheet_name=propshname)


def getpoints(li, points):
    #points={i: [] for i in range(10)}
    if li:
        for k in li:
            if k.isdigit():
                points[int(k)].append(i)
    return points

cnt=0
    
for index, col in enumerate(data_prop.columns):
    prop={i: [] for i in range(10)}
    gt={i: [] for i in range(10)}
    prev={i: [] for i in range(10)}
    bf={i: [] for i in range(10)}
    print(col)
    if col == "dummy" or col=="Unnamed: 0":
        continue
    for i in range(len(data_prop)):
        if "-1" in data_prop[col][i]:
            print("stopped at -1")
            break
        li_prop = list(data_prop[col][i].replace('[','').replace(']','').replace(' ', '').split(','))
        li_gt = list(data_gt[col][i].replace('[','').replace(']','').replace(' ', '').split(','))
        li_prev = list(data_prev[col][i].replace('[','').replace(']','').replace(' ', '').split(','))
        li_bf = list(data_bf[col][i].replace('[','').replace(']','').replace(' ', '').split(','))
        #print(li_bf)
        prop = getpoints(li_prop, prop)
        gt = getpoints(li_gt, gt)
        prev = getpoints(li_prev, prev)
        bf = getpoints(li_bf,bf)
        #print(bf)

    for c in range(10):
        plt.scatter(gt[c],[c] * len(gt[c]),c='r',marker='|',label="GT",s=400)
        plt.scatter(prop[c],[c] * len(prop[c]),c='b',marker='|', label="PROP",s=300)
        plt.scatter(prev[c],[c] * len(prev[c]),c='g',marker='|',label="PREV",s=100)
        plt.scatter(bf[c],[c] * len(bf[c]),c='y',marker='|',label="BF",s=50)
    plt.grid(True)
    plt.title("red: gt, blue: prop, green: prev, yellow:bf")
    #plt.legend(loc='upper right')
    plt.yticks((0,1,2,3,4,5,6,7,8,9))
    #plt.yticks(i for i in range(10), ['1','2','3','4','5','6','7','8','9','10'])

    #plt.show()
    plt.savefig("/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/graph/"+col+"_dt.png")
    plt.clf()
    cnt+=1
    # if cnt > 2:
    #     break
        

'''
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
    plt.plot(x, gt_list, 'ro',label="GT")
    plt.plot(x, bf_list, 'b-.', label="BF")
    plt.plot(x, prev_list, 'y:', label = "PREV")
    plt.plot(x, prop_list, 'g', label = "PROP")
    plt.legend(loc='upper right')
    plt.savefig("/home/spencer1/AIoTVirt/trajectoryanalysis/activation_graph/graph/"+col+".png")
    plt.clf()
    if cnt == 20:
        break
    cnt +=1

'''