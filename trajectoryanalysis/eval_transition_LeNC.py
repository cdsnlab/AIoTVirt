'''
this program finds the transition time between two camera with LENC. (spencer)
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


transition_map = {
    0: [1,8,9],
    1: [0, 8, 2],
    2: [1, 3, 7],
    3: [2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 7],
    7: [6, 3, 8],
    8: [0,1,7,9],
    9: [0,8]
}

def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

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


def averagetransitiontime(k,v):
    v = (sorted(v))
    #print(k, statistics.median(v))    
    return k, statistics.median(v)

def gettransitiontime(tmap, direction):
    for k in (tmap):
        if k == direction:
            return tmap[k]
    return -1

def stdoverlap(k, v):
    v = (sorted(v))
    #print(k, statistics.median(v))    
    return k, statistics.stdev(v)

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


def get_transition_dist(path, naive=False):
    filenames = os.listdir(path)
    transition = {}
    tmap={}
    atransition={}
    stransition={}
    for file in filenames:
        if '.csv' in file:
            #print(file)
            data = pd.read_csv(path + "/" + file)
            traces, cameras, labels = get_sequences(data)
            traces = sorted(traces, key=lambda t: t['start'])

            if not naive:
                traces = get_cam_order(traces)
                #print(traces)
            # get transition time (distribution)    
            for i, trace in enumerate(traces[1:]):
                #print(str(traces[i]['camera'])+"-->"+str(trace['camera']))
                if traces[i]["camera"] != trace["camera"]:
                    if traces[i]['end'] < trace['start']:
                        if str(traces[i]["camera"])+"-->"+str(trace["camera"]) not in transition:
                            transition[str(traces[i]['camera'])+"-->"+str(trace['camera'])] = [trace['start'] - traces[i]['end']]
                        else:
                            transition[str(traces[i]['camera'])+"-->"+str(trace['camera'])].append(trace['start'] - traces[i]['end'])


    for k, v in transition.items():
        if len(v) > 1:
            t, tt = averagetransitiontime(k,v)
            s, ss = stdoverlap(k,v)
            #print(tt)
            atransition[t]=tt
            stransition[s]=ss

    print (atransition)
    print (stransition)
    transition_results = pd.DataFrame(columns=["source_cam", "target_cam", "mse", "rmse", "mae"])
    name = 0
    predy=[]
    acty=[]
    '''
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
                predy.append(gettransitiontime(tmap, str(traces[i]["camera"])+"-->"+str(trace["camera"])))
                acty.append(trace['start'] - traces[i]['end'])
                transition_results = transition_results.append(pd.Series(data=[str(traces[i]["camera"]), str(trace["camera"]), mse(acty, predy), rmse(acty, predy), mae(acty, predy)], index=transition_results.columns, name=name))
            name+=1
    with pd.ExcelWriter("evaluation_transition_time_prev.csv") as writer:
        transition_results.to_excel(writer, sheet_name="previous approach")
    #return traces, cameras, labels, transition
    '''

argparser = argparse.ArgumentParser(
    description="welcome")
argparser.add_argument(
    '--path',
    metavar='p',
    help='tracks folder location'
)
args = argparser.parse_args()

#traces, cameras, labels, transition = 
get_transition_dist('/home/spencer1/samplevideo/train_new_sim_csv/', False)

