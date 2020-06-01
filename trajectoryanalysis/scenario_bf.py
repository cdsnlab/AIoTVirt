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
import requests
from openpyxl import load_workbook


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.InteractiveSession(config=config)

argparser = argparse.ArgumentParser(
    description="welcome")
argparser.add_argument(
    '--path',
    metavar='p',
    help='tracks folder location'
)
args = argparser.parse_args()

traces = {}

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def get_point(strpoint):
    point = strpoint.replace('(', '').replace(')', '').split(',')
    point = [int(p) for p in point]
    # return np.array(point)
    return point



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


def get_neighbour(grtraces):
    global transition_map
    final = []
    count = 0
    try:
        final.append(grtraces[0])
        for i, trace in enumerate(grtraces[1:-1]):
            src = final[-1]['camera']
            transitions = []
            for t in grtraces[len(final):]:
                if t['camera'] in transition_map[src]:
                    transitions.append(t)

            for nT in transitions:
                # if nT.camera in transition_map[src]:
                if nT['start'] > final[-1]['start'] and nT['end'] < final[-1]['end']:
                    continue
                if nT['start'] >= final[-1]['end']:
                    final.append(nT)
                    break
                else:
                    if final[-1]['end'] == nT['end']:
                        a = 0
                        continue
                    elif nT['end'] < final[-1]['end']:
                        continue
                    # final.append(Trace(final[-1].end, nT.end, nT.end - final[-1].end, nT.camera))
                    final.append({
                        "start": final[-1]['end'],
                        "end": nT['end'],
                        "duration": nT['end'] - final[-1]['end'],
                        "camera": nT['camera']
                    })
                    break
    except IndexError:
        pass

    return final

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
    print(k, statistics.median(v))    
    return k, statistics.median(v)


def get_transition_dist(path, naive=False):
    filenames = os.listdir(path)
    transition = {}
    tmap={}
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
                if str(traces[i]["camera"])+"-->"+str(trace["camera"]) not in transition:
                    transition[str(traces[i]['camera'])+"-->"+str(trace['camera'])] = [trace['start'] - traces[i]['end']]
                else:
                    transition[str(traces[i]['camera'])+"-->"+str(trace['camera'])].append(trace['start'] - traces[i]['end'])

    for k, v in transition.items(): # get transition time btw two locs.
        t, tt=averagetransitiontime(k,v)
        tmap[t]=tt
    return tmap


def get_correct(filedata):
    corr_traces, _, _ = get_sequences(filedata)
    corr_traces = sorted(corr_traces, key=lambda t: t['start'])
    corr_traces = get_neighbour(corr_traces)
    ground_truth = []
    for i, trace in enumerate(corr_traces[:-1]):

        cam = trace['camera']
        label = corr_traces[i + 1]['camera']
        ground_truth.append((trace['end'], cam, label))
    return ground_truth


for camera in range(10):
    loaded = np.load("/home/boyan/out_label_trans/{}.npy".format(camera), allow_pickle=True)
    traces[camera] = loaded

path ="/home/spencer1/samplevideo/sim_csv/prev_full/"
path1, dirs1, files1 = next(os.walk(path))
file_count = len(files1)
skip_file=0

filenames = os.listdir(path)
#filenames = filenames[::skip_file]

MIN_LENGTH = 15
MAX_SIM = 4000
results = {}
transitions = {}
possible_transitions = {} # possible camera and time.
rec_counter ={}
#possible_transitions = {i: -1 for i in range(10)}
container_boot_time = 0  # container boot time
tracker = 0
start_time = time.time()
#transition_times = get_transition_dist(path,False) # tmap
load_tmap_time = time.time()
print("loaded tmap in {} seconds".format(load_tmap_time-start_time))
time.sleep(2)
name=0 # for excel file



results_for_excel = pd.DataFrame(
    columns = ["filename", "percentage of frames processed / total number of frames", "percentage of target frames / total number of frames", "precision", "accuracy"]
)

activation_results = pd.DataFrame(
    columns = ["dummy"]
)
gt_activation_results = pd.DataFrame(
    columns = ["dummy"]
)

for file in tqdm(filenames):
    if '.csv' not in file:
        continue
    print(file)
    file_start_time=time.time()
    data = pd.read_csv(path + file)
    rowsize = len(data['Camera 1'])
    #print(rowsize)
    cam_tracking = -1

    # data = pd.read_csv(args.path)
    handover_points = []
    
    recording = {i: [] for i in range(10)}
    processed_frames = {i: 0 for i in range(10)}
    cnt_seen_target= {i: 0 for i in range(10)}
    number_of_activated_cameras = {i: 0 for i in range(MAX_SIM)}
    gt_activated = {i: 0 for i in range(MAX_SIM)}

    available_states = ["rec", "trk", "trans"]
    state = "rec" # inital state
    cnt_ = {key: 0 for key in available_states}
    cnt_total = 0 #* total number of frames.
    cnt_target =0 #* howmany valid frames there are in this file
    sump=0
    sumst=0
    for index, row in data.iterrows(): #* read frame by frame.
        # * Spencers approach * #
        cnt_[state]+=1
        cnt_total+=1
        
        for camera in range(10):
            value = row['Camera {}'.format(camera)]
            if '-1' not in value:
                cnt_target+=1
                gt_activated[index]+=1
                break
        
        if state=="rec": #* recovery: turn on all cameras to search for the target.
            print("[INFO] in REC at frame number {}".format(index))
            for camera in range(10):
                value = row['Camera {}'.format(camera)]
                processed_frames[camera] += 1
                number_of_activated_cameras[index]+=1
                if '-1' not in value:
                    # * this is a new trk, 1) add new trk, 2) exit recovery mode.
                    cnt_seen_target[camera]+=1
                    recording[camera].append({
                        "start": index,
                        "duration": 0,
                        "tracks": [get_point(value)],
                        "end": -1,
                    })
                    cam_tracking=camera
                    state="trk"
                    break

        elif state == "trk": #* tracking: 
            processed_frames[cam_tracking] +=1
            number_of_activated_cameras[index]+=1
            # * We have a camera that is tracking, continue here
            camera = cam_tracking
            print("[INFO] in TRK in camera {} frame number {}".format(camera, index))

            value = row['Camera {}'.format(camera)]
            # * Increase the current trace's duration
            recording[camera][-1]['duration'] += 1
            if '-1' in value: # * If person not here, end current trace
                # handover_points.append((index, cam_tracking, perform_handover[1]))
                recording[cam_tracking][-1]['end'] = index
                prev_tracking = cam_tracking
                cam_tracking = -1
                state="trans"
                container_boot_time = 15 #* this means that it suffers 15 bootup time for all other cameras.

            else: # * If person is here, continue recording trace
                state="trk"
                recording[camera][-1]['tracks'].append(get_point(value))
                cnt_seen_target[camera]+=1
                

        elif state == "trans": #* in between cams
            #print("[INFO] in TRANS at frame number {}".format(index))
            # 1) look at each camera one by one.
            if container_boot_time ==0:
                print("[INFO] in TRANS going to REC at frame number {}".format(index))
                state="rec"
                continue
            else:
                print("[INFO] in TRANS:CONTAINER_BOOTING at frame number {}".format(index))
            container_boot_time -= 1
        if index==rowsize-1:
            for i in range(rowsize-1, MAX_SIM):
                number_of_activated_cameras[i] = -1
                gt_activated[i] = -1
    #? calculate results for each iteration. 

    for k, v in processed_frames.items():
        sump += v
    for k, v in cnt_seen_target.items():
        sumst += v
    print("percentage of frames processed / total number of frames {}".format(100*sump/(cnt_total*10))) #* processed frames from the vid
    print("percentage of target / total number of frames {}".format(100*cnt_target/(cnt_total*10))) #* target frames in the video
    print("precision {}".format(100*sumst/sump)) #* precison 
    if cnt_target ==0:
        print("accuracy {}".format(0)) #* accuracy
        results_for_excel = results_for_excel.append(pd.Series(data=[file, 100*sump/(cnt_total*10), 100*cnt_target/(cnt_total*10), 100*sumst/sump, 0], index=results_for_excel.columns, name=name))
    else:
        print("accuracy {}".format(100*sumst/cnt_target)) #* accuracy
        results_for_excel = results_for_excel.append(pd.Series(data=[file, 100*sump/(cnt_total*10), 100*cnt_target/(cnt_total*10), 100*sumst/sump, 100*sumst/cnt_target], index=results_for_excel.columns, name=name))

    activation_results[file[:-4]+"_activenumber"] = pd.Series(number_of_activated_cameras)
    gt_activation_results[file[:-4]+"_activenumber"] = pd.Series(gt_activated)
    name+=1
    file_end_time = time.time()
    print("file iteration time {}".format(file_end_time - file_start_time))


sim_end_time = time.time()
print("[INFO] total time {}".format(sim_end_time - start_time))        
with pd.ExcelWriter("results/scenario_results_bf.xlsx", mode='a') as writer:
    results_for_excel.to_excel(writer, sheet_name="bf-test-activation-of-cameras")

with pd.ExcelWriter("activation_graph/bf_activation.xlsx", mode='a') as writer:
    activation_results.to_excel(writer, sheet_name="test-activation_number")

# with pd.ExcelWriter("activation_graph/gt_activation.xlsx", mode='a') as writer:
#     gt_activation_results.to_excel(writer, sheet_name="test-activation_number")
print("[INFO] DONE! ")