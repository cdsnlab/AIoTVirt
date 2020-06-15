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
transition_map = {
    0: [1, 2, 9],
    1: [0, 2, 8],
    2: [1, 3, 7],
    3: [2, 4],
    4: [2, 3, 5],
    5: [4, 6],
    6: [5, 7, 8],
    7: [2, 3, 6],
    8: [1, 2, 9],
    9: [0, 6, 9]
}

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


def get_point(strpoint):
    point = strpoint.replace('(', '').replace(')', '').split(',')[0:2]
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


def get_transition_dist(path, naive=False):
    filenames = os.listdir(path)
    transition = {}
    tmapmin, tmapmax={}, {}
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

        v = (sorted(v))
        if min(v) < 0:
            tmapmin[k]=0
        else:
            tmapmin[k]=min(v)
        if max(v) < 0:
            tmapmax[k]=0
        else:
            tmapmax[k]=max(v)
    return tmapmin, tmapmax


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


# for camera in range(10):
#     loaded = np.load("/home/boyan/out_label_trans/{}.npy".format(camera), allow_pickle=True)
#     traces[camera] = loaded

#path ="/home/spencer1/samplevideo/full_old_sim_csv/prev_full/"
path ="/home/spencer1/samplevideo/test_new_sim_csv/"
path_full = "/home/spencer1/samplevideo/train_new_sim_csv/"

filenames = os.listdir(path)
#filenames = filenames[::11]
#skip_file=3000
#filenames = filenames[1::skip_file]
shname = "fixed_waiting_time"
MIN_LENGTH = 15
MAX_SIM = 4000
results = {}
transitions = {}
possible_transitions = {} # possible camera and time.
cnt_container={}
rec_counter ={}
#possible_transitions = {i: -1 for i in range(10)}
container_boot_time = 0  # container boot time
tracker = 0
start_time = time.time()
transition_min, transition_max = get_transition_dist(path_full,False) # tmap
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


for file in tqdm(filenames):
    if '.csv' not in file:
        continue
    print(file)
    file_start_time = time.time()
    data = pd.read_csv(path + file)
    rowsize = len(data['Camera 1'])
    print(rowsize)

    cam_tracking = -1

    perform_handover = (-1, -1)  # * Frame index; Camera
    # data = pd.read_csv(args.path)
    gr_truth = get_correct(data)
    handover_points = []
    
    recording = {i: [] for i in range(10)}
    processed_frames = {i: 0 for i in range(10)}
    cnt_seen_target= {i: 0 for i in range(10)}
    number_of_activated_cameras = {i: 0 for i in range(MAX_SIM)}

    actaivated_camera_indexs = {i: [] for i in range(MAX_SIM)}
    
    available_states = ["rec", "trk", "trans"]
    state = "rec" # inital state
    cnt_ = {key: 0 for key in available_states}
    
    cnt_total = 0 #* total number of frames.
    cnt_target =0 #* how many valid frames there are in this file
    sump=0
    sumst=0
    trktimer=0
    begin=True
    for index, row in data.iterrows(): #* read frame by frame.
        # * Spencers approach * #
        cnt_[state]+=1
        cnt_total+=1
        
        for camera in range(10):
            value = row['Camera {}'.format(camera)]
            if '-1' not in value:
                cnt_target+=1
                break
        
        if state=="rec": #* recovery: turn on all cameras to search for the target.
            print("[INFO] in REC at frame number {}".format(index)) #! this is the reason for low precision
            for camera in range(10):
                if begin != True:
                    processed_frames[camera] += 1
                    number_of_activated_cameras[index]+=1
                    actaivated_camera_indexs[index].append(camera)
                value = row['Camera {}'.format(camera)]
                
                if '-1' not in value:
                    # * this is a new trk, 1) add new trk, 2) exit recovery mode.
                    recording[camera].append({
                        "start": index,
                        "duration": 0,
                        "tracks": [get_point(value)],
                        "end": -1,
                    })
                    cam_tracking=camera
                    state="trk"
                    cnt_seen_target[camera]+=1
                    begin=False
                    
                    break

        elif state == "trk": #* tracking: 
            camera = cam_tracking

            processed_frames[camera] +=1
            number_of_activated_cameras[index]+=1
            actaivated_camera_indexs[index].append(camera)
            # * We have a camera that is tracking, continue here
            print("[INFO] in TRK in camera {} frame number {}".format(camera, index))
            value = row['Camera {}'.format(camera)]
            # * Increase the current trace's duration
            recording[camera][-1]['duration'] += 1
            if '-1' in value: # * If person not here, end current trace
                trktimer+=1

                for i in transition_map[camera]: # get neighboring cameras transition diff.
                    if i != camera: #! maybe set a minimum transition time for possible transitions and wait until it finds it?
                        print("[INFO] possible transition from {} to {} after {} frames".format(camera, i, index+container_boot_time))
                        #possible_transitions[i] = int(transition_min[str(camera)+"-->"+str(i)])-container_boot_time
                        #possible_transitions[i] = int(transition_min[str(camera)+"-->"+str(i)])+index
                        possible_transitions[i] = index+container_boot_time
                        #rec_counter[i]=0
                        rec_counter[i]= int(transition_max[str(camera)+"-->"+str(i)])
                        cnt_container[i]=15
                if trktimer > 15: #! waits couple frames until saying its gone.
                    print("===[INFO] Fcuk i m out")
                    recording[cam_tracking][-1]['end'] = index
                    prev_tracking = cam_tracking
                    cam_tracking = -1
                    
                    state="trans"
                    trktimer=0
                
            else: # * If person is here, continue recording trace
                cnt_seen_target[camera]+=1
                recording[camera][-1]['tracks'].append(get_point(value))
                trktimer=0
                
        elif state == "trans": #* in between cams
            print("[INFO] in TRANS at frame number {}".format(index))

            if not possible_transitions:
                state = "rec"
                continue
            for k, v in possible_transitions.items():
                print("[INFO] possible transition point from camera {} to camera {} at {}th frame".format(prev_tracking, k, v))
                if v <= index: #* check if transition index is earlier than current index.                    
                    camera=k
                    value = row['Camera {}'.format(camera)]
                    cnt_container[k] -= 1
                    if cnt_container[k] <= 0: #* CONTAINER BOOT TIME IS DONE! search in these cameras 
                        processed_frames[camera] += 1
                        number_of_activated_cameras[index]+=1
                        actaivated_camera_indexs[index].append(camera)
                        if '-1' not in value:
                            # * correct transfer.
                            print("[INFO] found target in camera {} in {}th frame".format(camera, index))
                            cnt_seen_target[camera]+=1

                            recording[camera].append({
                                "start": index,
                                "duration": 0,
                                "tracks": [get_point(value)],
                                "end": -1,
                            })
                            state="trk"
                            cam_tracking=camera
                            possible_transitions={} #* clear the transition points
                            break
                        print(k, rec_counter[k])
                        rec_counter[k]-=1 #* to show if there is an object here.
                    if rec_counter[k] < 0: #* give 15 frames until calling missing.
                        print("[INFO] pop transition destination for {} at {}th frame".format(k, index))
                        possible_transitions.pop(k)
                        cnt_container.pop(k)
                        break
        #print(index, rowsize-2)
        if index==rowsize-2:
            for i in range(rowsize-2, MAX_SIM):
                number_of_activated_cameras[i] = -1
                actaivated_camera_indexs[i].append(-1)

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
    activation_results[file[:-4]] = pd.Series(actaivated_camera_indexs)

    name+=1
    file_end_time = time.time()
    print("file iteration time {}".format(file_end_time - file_start_time))
sim_end_time = time.time()
print("[INFO] total time {}".format(sim_end_time - start_time))
####! logfiles
with pd.ExcelWriter("results/scenario_results_newsimdata_prev.xlsx", mode='a') as writer:
    results_for_excel.to_excel(writer, sheet_name=shname)

with pd.ExcelWriter("activation_graph/prev_newsimdata_activation.xlsx", mode='a') as writer:
    activation_results.to_excel(writer, sheet_name=shname)
print("[INFO] DONE! ")