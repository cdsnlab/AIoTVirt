'''
this program shows how much overlap there is between two cameras in a given set of CSV.
you are a dick head. if the value is - its an overlap, if its + its a blindspot...
'''

import json
import time

import numpy as np
import pandas as pd
import argparse, math
from scipy import stats
import statistics
#from keras.models import load_model
#import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#from termcolor import cprint
import requests
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import pandas as pd

def averageoverlap(k,v):
    v = (sorted(v))
    #print(k, statistics.median(v))
    #print("%.2f"%(statistics.median(v)))    
    return k, "%.2f"%(statistics.median(v)/15)

def stdoverlap(k, v):
    v = (sorted(v))
    #print(k, statistics.median(v))    
    return k, "%.2f"%(statistics.stdev(v)/15)



# Original - Least number of cameras and full coverage
def get_cam_order(traces):
    for i, trace in enumerate(traces):
        for other in traces[i:]:
            if other['start'] > trace['start'] and other['end'] <= trace['end']:
                traces.remove(other)
    i = 1
    to_remove = []
    for trace in traces[1:-1]:
        try:
            if trace['start'] < traces[i - 1]['end'] and trace['end'] > traces[i + 1]['start'] and traces[i - 1]['end'] > traces[i + 1]['start']:
                traces.remove(trace)
                i -= 1
            i += 1
        except IndexError:
            print(trace)
            pass

    return traces

def get_sequences(data: pd.DataFrame):
    seq_traces = []

    labels = []
    cameras = {}
    #print(data)
    for i in range(10):
        camera = []
        current_sequence = 0
        start = 0
        labels.append('Camera {}'.format(i))
        if 'Camera {}'.format(i) in data:
            for end, value in data['Camera {}'.format(i)].items():
                if '-1' not in value:
                    if current_sequence == 0:
                        start = end
                    current_sequence += 1
                else:
                    if current_sequence != 0 and current_sequence > 15:
                        camera.append((start, current_sequence))
                        seq_traces.append({
                            "start": start,
                            "end": end,
                            "duration": current_sequence,
                            "camera": i
                        })
                    current_sequence = 0
            if current_sequence != 0 and current_sequence > 15:
                camera.append((start, current_sequence))
                # seq_traces.append(Trace(start, end, current_sequence, i))
                seq_traces.append({
                            "start": start,
                            "end": end,
                            "duration": current_sequence,
                            "camera": i
                        })
            cameras[i] = camera
    return seq_traces, cameras, labels

        

def get_overlaps(path, naive=False):
    filenames = os.listdir(path)
    overlap = {}
    aoverlap={}
    soverlap={}
    for file in filenames:
        if '.csv' in file:
            #print(file)
            data = pd.read_csv(path + "/" + file)
            traces, cameras, labels = get_sequences(data)
            traces = sorted(traces, key=lambda t: t['start'])

            if not naive:
                traces = get_cam_order(traces)
                #print(traces)
            for i, trace in enumerate(traces[1:]): 
                #if traces[i]['end'] > trace['start']:
                if traces[i]["camera"] != trace["camera"]:
                    if str(traces[i]['camera'])+"-"+str(trace['camera']) not in overlap:
                        overlap[str(traces[i]['camera'])+"-"+str(trace['camera'])] = [trace['start'] - traces[i]['end']]
                    else:
                        overlap[str(traces[i]['camera'])+"-"+str(trace['camera'])].append(trace['start'] - traces[i]['end'])

        
    for k, v in overlap.items():
        if len(v) > 1:
            #print(v)
            t, tt = averageoverlap(k,v)
            s, ss = stdoverlap(k,v)
            #print(tt)
            aoverlap[t]=tt
            soverlap[s]=ss

    print (aoverlap)
    print (soverlap)
    
    
    name = 0
    transition_results = pd.DataFrame(columns=["cam pair", "avg time", "stdev"])
    for k, v in aoverlap.items():
        print(k +" & " + str(v) +" & " + str(soverlap[k]) +" \\\ \hline")
        transition_results = transition_results.append(pd.Series(data=[k, str(v), str(soverlap[k])], index=transition_results.columns, name=name))
        name+=1
    with pd.ExcelWriter("evaluation_transition_time_new_sim.xlsx", mode='w') as writer:
        transition_results.to_excel(writer, sheet_name="overlap & blindspot")
    '''
    predy=[]
    acty=[]
    
    for file in filenames:
        if '.csv' in file:
            print(file)
            data = pd.read_csv(path + "/" + file)
            traces, cameras, labels = get_sequences(data)
            traces = sorted(traces, key=lambda t: t['start'])

            if not naive:
                traces = get_cam_order(traces)

            for i, trace in enumerate(traces[1:]):
                # if start point is 0 and end points are in any of 1, 8, 9. record mse, mae, rmse
                predy.append(averageoverlaps(tmap, str(traces[i]["camera"])+"-->"+str(trace["camera"])))
                acty.append(trace['start'] - traces[i]['end'])
                transition_results = transition_results.append(pd.Series(data=[str(traces[i]["camera"]), str(trace["camera"]), mse(acty, predy), rmse(acty, predy), mae(acty, predy)], index=transition_results.columns, name=name))
            name+=1
    
    with pd.ExcelWriter("evaluation_transition_time_prev.csv") as writer:
        transition_results.to_excel(writer, sheet_name="previous approach")
    '''
    #return traces, cameras, labels, transition



argparser = argparse.ArgumentParser(
    description="welcome")
argparser.add_argument(
    '--path',
    metavar='p',
    help='tracks folder location'
)
args = argparser.parse_args()

#traces, cameras, labels, transition = 
get_overlaps('/home/spencer1/samplevideo/new_sim_csv/', False)
