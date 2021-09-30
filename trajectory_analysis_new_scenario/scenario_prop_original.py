import json
import time

import numpy as np
import pandas as pd
import argparse, math
from scipy import stats
import statistics
from keras.models import load_model
import tensorflow as tf
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests
import csv
import pickle
import joblib
import argparse


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

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
    0: [0, 1, 2, 9],
    1: [0, 1, 2, 8],
    2: [1, 2, 3, 7],
    3: [2, 3, 4],
    4: [2, 3, 4, 5],
    5: [4, 5, 6],
    6: [5, 6, 7, 8],
    7: [2, 3, 6, 7],
    8: [1, 2, 8, 9],
    9: [0, 8, 9]
}


#models_time = {cam: load_model('/home/boyan/AIoTVirt/trajectoryanalysis/models/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
#models_time = {cam: load_model('/home/boyan/AIoTVirt/trajectoryanalysis/models/cam_sw_o_{}.h5'.format(cam), compile=False) for cam in range(10)}
models_time = {cam: load_model('/home/boyan/AIoTVirt/trajectoryanalysis/models/mae_cam_last_resnet_1000_{}.h5'.format(cam), compile=False) for cam in range(10)}
models = {cam: load_model('/home/spencer1/AIoTVirt/trajectoryanalysis/models/connected_new_sim/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
#models = {cam: load_model('/home/spencer1/AIoTVirt/trajectoryanalysis/model_new_sim/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
#models = {cam: load_model('/home/boyan/LSTM_Cam_pred/model_100epochs/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
#models = {cam: load_model('/home/boyan/LSTM_Cam_pred/model_new_label/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
#dtmodel = {cam: joblib.load('/home/spencer1/AIoTVirt/trajectoryanalysis/joblibs/dt/swo_duration_transition_{}.joblib'.format(cam))for cam in range(10)}
dtmodel = {cam: joblib.load('/home/spencer1/AIoTVirt/trajectoryanalysis/joblibs/dt/last_duration_transition_{}.joblib'.format(cam))for cam in range(10)}

passed_frames = 0

for camera in range(10):
    #loaded = np.load("/home/boyan/out_label_trans/{}.npy".format(camera), allow_pickle=True)
    #loaded = np.load("/home/boyan/out_label_neighb_irw/{}.npy".format(camera), allow_pickle=True)
    # loaded = np.load("/home/boyan/out_label_neighb/{}.npy".format(camera), allow_pickle=True)
    #loaded = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/npy/train_multi_output/{}.npy".format(camera), allow_pickle=True)
    loaded = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/npy/connected/{}.npy".format(camera), allow_pickle=True)
    traces[camera] = loaded


def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rads = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return math.degrees(rads)


def translate_vector(vec):
    start = vec[0]
    end = vec[1]
    new_end = end - start
    return new_end


def get_point(strpoint):
    point = strpoint.replace('(', '').replace(')', '').split(',')[0:2]
    point = [int(p) for p in point]
    # return np.array(point)
    return point


def estimate_handover(source, dest, cur_path, len_cur_path, angle_lim=30, cutoff_long=False, cutoff_short=False):
    remaining_durations = []
    remaining_transitions = []
    cur_dist = 0
    for i, point in enumerate(cur_path[:-1]):
        cur_dist += math.hypot(point[0] - cur_path[i + 1][0], point[1] - cur_path[i + 1][1])

    a = cur_path
    b = cur_path[0]
    c = cur_path[-1]
    cur_vector = translate_vector((cur_path[0], cur_path[-1]))
    if dest == -1:
        return -1, 0
    for path, label, length, transition in traces[source][:-1]:
        if label == dest:
            # * Calculate current target trace distance
            # ? Can we calculate it only the first time we encounter the trace?
            dist = 0
            for i, point in enumerate(path[:-1]):
                dist += math.hypot(point[0] - path[i + 1][0], point[1] - path[i + 1][1])

            # * Calculate approximate remaining distance
            end = path[-1]
            cur_point = cur_path[-1]
            min_remaining = math.hypot(end[0] - cur_point[0], end[1] - cur_point[1])

            # * Remove paths shorter than ours already is in either time or distance
            if len(path) < len_cur_path or dist < cur_dist:
                continue
            # * Remove paths that have a distance shorter than our minimum possible distance
            if cutoff_short and 0.95 * (cur_dist + min_remaining) > dist:
                continue
            # # * If min_remaining relatively small AND len difference between cur_path and path too large, discard point?
            # # * Logic being yes, we are close to the exit but the other path took the trace very slowly
            # # ? Can this be improved?
            # # ? Specifically, in calculation of distance to path relations
            if cutoff_long and min_remaining / cur_dist < 0.20:
                if len_cur_path / len(path) < 1 - min_remaining / cur_dist:
                    continue
            # * Calculate angle between current and existing path
            # * If angle is too large (above given threshold) then the existing trace has a different heading
            path_vector = translate_vector((path[0], path[-1]))
            angle = angle_between(cur_vector, path_vector)
            if angle > angle_lim:  # ? Angle between vectors is always <180 or angle < -angle_lim:
                continue
            remaining_durations.append(len(path))
            remaining_transitions.append(transition)
    # ! DEAL WITH THE CASE WHERE EVERYTHING IS FILTERED OUT
    if len(remaining_durations) != 0:
        # TODO Calculate mean transition distance
        #print("[DEBUG] for {} -> {} ".format(source, dest))
        # return stats.mode(remaining_durations)[0][0], stats.mode(remaining_transitions)[0][0]
        return int(statistics.median(remaining_durations)), int(statistics.median(remaining_transitions))
    else:
        #print("[DEBUG] Nothing for {} -> {} ".format(source, dest))
        return -1, 0


def preprocess_track(track, vl): #*ED
    # if len(track) < vl:
    #     continue
    # TODO NORMALIZE THIS FUCKING TRACK

    output = []
    btw = math.floor(len(track) / float(vl))
    for x, y in track[::btw]:
        #print("[DEBUG] ", x, y)
        if len(output) == vl:
            break
        else:
            output.append((x / 2560, y / 1440))
    return np.array([output])

def preprocess_track_last(track, vl): #*last
    
    output = []
    btw = math.floor(len(track) / float(vl))
    for x,y in track:
        output.append(np.array([x / 2560, y / 1440]))
    #print(output[-vl:])
    return np.array([output[-vl:]])



threshold = 30
enable_filter=False
not_time=False


def resnet_handover(src_camera, full_track, vl=30, sample_method="last"):
    if sample_method=="last":
        track=preprocess_track_last(full_track,vl)
        #track=np.array([full_track[-vl:]])
    else: # ed
        track=preprocess_track(full_track, vl)
    #s, nx, ny = track.shape
    #track = track.reshape((s, nx*ny))
    #estimation = int(dtmodel[src_camera].predict(track))
    estimation = models_time[src_camera].predict(track)
    print(int(estimation))
    #return int(models_time[src_camera].predict(track))
    return int(estimation)
    

def estimate_transition():
    global perform_handover, transition, models, cam_tracking, passed_frames, enable_filter
    # transition[camera][-1]['tracks'].append(get_point(value))
    if passed_frames % 5 == 0:
        if recording[camera][-1]['duration'] > threshold:
            choice=[]
            handover_time=[]
            transition_duration=[]
            tracks = recording[camera][-1]['tracks']
            full_track = np.array(tracks)
            len_tracks = len(tracks)
            track = preprocess_track(tracks, 30)
            predicted_camera = models[camera].predict(track)
            p = predicted_camera[0].argsort()
            #print("First option {}: {}, second option {}: {}".format(predicted_camera[0][p[-2:][::-1][0]], p[-2:][::-1][0], predicted_camera[0][p[-2:][::-1][1]], p[-2:][::-1][1])) #* p[-a:][::-b][::-c] a:how many predictions, b:? c: which one do u want
            for i in range(2):
                if p[-2:][::-1][i] != camera:
                    if predicted_camera[0][p[-2:][::-1][i]]>0.4: #! this is cuz all classes must add up to 1. 
                        #print("[DEBUG] added {} as {} option".format(p[-2:][::-1][i], i ))
                        choice.append(p[-2:][::-1][i])
                    else:
                        choice.append(-1)
     

            for i, target in enumerate (choice):
                #* handover_t: how long the trace is.., transition_d: how long it took from camera a to b
                if target != -1:
                    if not enable_filter:
                        handover_td = resnet_handover(camera, full_track,30, "last")
                        handover_t, transition_d = handover_td, 0
                    elif not_time:
                        handover_t, transition_d = 0,0
                    else:
                        handover_t, transition_d = estimate_handover(camera, target, full_track, len_tracks, 10, False, False)
                    
                    #print(target, handover_t, transition_d)
                    #handover_time.append(handover_t)
                    #transition_duration.append(transition_d)
                    #print("[DEBUG] destination {}, handover_time {}, transition_dur {}".format(k, handover_t, transition_d)) 
                    # TODO Adjust handover time based on threshold and container_boot_time
                    if handover_t!=-1:
                        if not enable_filter:
                            #print(handover_t, transition_d, recording[camera][-1]['start'])
                            
                            #perform_h = (handover_t + transition_d - container_boot_time +recording[camera][-1]['start'], target) 
                            perform_h = (index+handover_t-container_boot_time, target) 
                        else:
                            perform_h = (handover_t + transition_d + index - container_boot_time - recording[camera][-1]['duration'], target) # - recording[camera][-1]['duration'], target)
                        
                        perform_handover[i]=(perform_h)

                        # * As we are tracking in this camera, we don't need to check the rest
                        cam_tracking = camera
    passed_frames += 1



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

#path ="/home/spencer1/samplevideo/sim_csv/prev_full/"
path ="/home/spencer1/samplevideo/test_new_sim_csv/"
path_full = "/home/spencer1/samplevideo/train_new_sim_csv/"

path1, dirs1, files1 = next(os.walk(path))
file_count = len(files1)
skip_file=1
#shname="even"
shname="resnet_2000_last"
filenames = os.listdir(path)
#filenames = filenames[1::skip_file]
filenames = filenames[::skip_file]
column_names=[]
for i in range(int(file_count/skip_file)):
    column_names.append("gt_"+str(i))
    column_names.append("prop_"+str(i))


MIN_LENGTH = 15
MAX_SIM=4000
results = {}
transitions = {}
possible_transitions = {} # possible camera and time.
rec_counter ={}
#possible_transitions = {i: -1 for i in range(10)}
container_boot_time = 15  # container boot time
tracker = 0
start_time = time.time()

#transition_times = get_transition_dist(path,False) # tmap
transition_min, transition_max = get_transition_dist(path_full,False) # tmap


#load_tmap_time = time.time()
#print("loaded tmap in {} seconds".format(load_tmap_time-start_time))
time.sleep(2)
name=0 # for excel file
results_for_excel = pd.DataFrame(
    columns = ["filename", "percentage of frames processed / total number of frames", "percentage of target frames / total number of frames", "precision", "accuracy"]
)

camera_evaluation_results = pd.DataFrame(
    columns = column_names
)

activation_results = pd.DataFrame(
    columns = ["dummy"]
)
early_late_results = pd.DataFrame(
    columns = ["dummy"]
)
valutearly={}
valutwrong={}
valutmissing={}
summ=0
for file in tqdm(filenames):
    valutmissing[file] = 0
    valutearly[file]=0
    valutwrong[file]=0
    if '.csv' not in file:
        continue
    print(file)
    file_start_time=time.time()
    data = pd.read_csv(path + file)
    rowsize = len(data['Camera 1'])

    wrong = 0
    early = 0
    missing = 0
    cam_tracking = -1
    perform_handover = {0: (-1, -1), 1: (-1,-1)}  # * Frame index; Camera
    # data = pd.read_csv(args.path)
    number_of_activated_cameras = {i: 0 for i in range(MAX_SIM)}
    recording = {i: [] for i in range(10)}
    processed_frames = {i: 0 for i in range(10)}
    cnt_seen_target= {i: 0 for i in range(10)}

    actaivated_camera_indexs = {i: [] for i in range(MAX_SIM)}

    cnt_camera_choice ={}
    cnt_camera_ans ={}
    available_states = ["rec", "trk", "trans", "a-rec"]
    state = "rec" # inital state
    cnt_ = {key: 0 for key in available_states}
    cnt_total= 0 #* total number of frames.
    cnt_target=0 #* howmany valid frames there are in this file
    cnt_rec=0 #* how long you were in REC state
    cnt_trk=0 #* how long you were in TRK state
    cnt_trans=0 #* how long you were in TRANS state
    sump=0
    sumst=0
    trktimer=0
    begin=True
    for index, row in data.iterrows(): #* read frame by frame.
        # * Spencers approach * #
        cnt_[state]+=1
        cnt_total+=1

        if index==rowsize-2:
            for i in range(rowsize-2, MAX_SIM):
                number_of_activated_cameras[i] = -1
                actaivated_camera_indexs[i].append(-1)
        #* to see if transition is wrong or time is wrong.
        cnt_camera_=[]
        for camera in range(10):
            value = row['Camera {}'.format(camera)]
            if '-1' not in value:
                cnt_camera_.append(camera)
        if not cnt_camera_:
            cnt_camera_.append(-1)
        cnt_camera_ans[index] = cnt_camera_   
        
        for camera in range(10):
            value = row['Camera {}'.format(camera)]
            if '-1' not in value:
                cnt_target+=1
                break
        
        if state=="rec": #* recovery: turn on all cameras to search for the target.
            cnt_rec+=1
            print("[INFO] in REC at frame number {}".format(index))

            cnt_camera_choice[index] = [-1]
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
                    cnt_camera_choice[index] = [camera]
                    begin=False
                    break

        elif state == "trk": #* tracking: 
            cnt_trk+=1
            camera = cam_tracking

            processed_frames[camera] +=1
            number_of_activated_cameras[index]+=1
            actaivated_camera_indexs[index].append(camera)

            value = row['Camera {}'.format(camera)]
            missing = 0
            early=0
            wrong = 0

            if '-1' in value: # * If person not here, end current trace
                print("+++[TRK] should be here {} frame number {}".format(camera, index))
                trktimer+=1
                
                if trktimer > 15:
                    print("===[TRK] Fcuk i m out")
                    recording[cam_tracking][-1]['end'] = index
                    prev_tracking = cam_tracking
                    cnt_camera_choice[index] = [-1]
                    #cam_tracking = -1
                    state="trans" 
                    trktimer=0
                
            else: # * If person is here, continue recording trace
                # * We have a camera that is tracking, continue here
                
                cnt_camera_choice[index] = [cam_tracking]
                cnt_seen_target[camera]+=1
                recording[camera][-1]['tracks'].append(get_point(value))
                print("[TRK] in TRK in camera {} frame number {}".format(camera, index))
 
                # * Increase the current trace's duration
                recording[camera][-1]['duration'] += 1
                trktimer=0
                            
            estimate_transition() #! check if the index of the prediction matches current index

            for k, v in perform_handover.items(): #! regardless of if the person is there or not, keep updating possible transitions.
                print("[TRK] these are possible transitions {}".format(v))
                if v[1] != -1:

                    possible_transitions[k] = v
                    rec_counter[k] = 0
               
        elif state == "trans": #* in between cams
            # ! can we know if its there but they are missing it? 
            el_gt = []
            el_p = []
            for i in range(10):
                value = row['Camera {}'.format(i)]
                if '-1' not in value:
                    el_gt.append(i)
                            
            cnt_trans+=1
            cnt_camera_choice[index]=[-1]
            print("[TRANS] in TRANS at frame number {}".format(index))
            if not possible_transitions:
                print("[TRANS] going to A-REC")
                for i in transition_map[prev_tracking]: # get neighboring cameras transition diff.
                    if i != prev_tracking: #! maybe set a minimum transition time for possible transitions and wait until it finds it?
                        print("[TRANS] possible transition from {} to {} after {} frames".format(prev_tracking, i, transition_min[str(prev_tracking)+"-->"+str(i)]))
                        #possible_transitions[i] = int(transition_min[str(camera)+"-->"+str(i)])-container_boot_time
                        possible_transitions[i] = (int(transition_min[str(prev_tracking)+"-->"+str(i)])+index, i)
                        #rec_counter[i]=0
                        rec_counter[i]=60
                state = "a-rec"
                el_p.append(-1)
                continue
            for k, v in possible_transitions.items():
                if v[0]<=index: #transition time == index #? IT ONLY CHECKS when the predictions are smaller than current index!
                    print("[TRANS] possible transition point from camera {} to camera {} at {}th frame".format(prev_tracking, v[1], v[0]))
                    number_of_activated_cameras[index]+=1
                    camera=v[1]
                    actaivated_camera_indexs[index].append(camera)
                    processed_frames[camera] += 1
                    value = row['Camera {}'.format(camera)]
                    el_p.append(camera)
                    if '-1' not in value:
                        # * correct transfer.
                        print("[TRANS] found target in camera {} in {}th frame".format(camera, index))
                        cnt_seen_target[camera]+=1
                        
                        recording[camera].append({
                            "start": index,
                            "duration": 0,
                            "tracks": [get_point(value)],
                            "end": -1,
                        })
                        state="trk"
                        cam_tracking=camera
                        perform_handover = {0: (-1, -1), 1: (-1,-1)}
                        possible_transitions = {} #* clear possible transition history.
                        cnt_camera_choice[index]=[camera]
                        
                        break
                    rec_counter[k]+=1
                else: #! what if they are far ahead? do nothing until it reaches that number.
                    print("[TRANS] waiting until {}th frame to camera {}".format(v[0], v[1]))
                    el_p.append(-1)
          
                if rec_counter[k] >= 60: #* give 30 frames until calling missing.
                    print("[TRANS] pop transition destination for {} at {}th frame".format(k, index))
                    possible_transitions.pop(k)
                    break
            #! check ehre
            #print("==================missing counter {} elgt {}, elp {}".format(missing, el_gt, el_p))
            if el_gt and int(-1) in el_p: #* late!
                #print(" ======= {} late counter {} elgt {}, elp {}".format(index, missing, el_gt, el_p))
                if missing > valutmissing[file]:
                    valutmissing[file] = missing
                missing+=1
            elif el_gt and el_p:
                for i in el_p:
                    if i not in el_gt:
                        #print(" ------ {} wrong prediction {}, {}".format(index, el_gt, el_p))
                        if wrong > valutwrong[file]:
                            valutwrong[file] = wrong
                        wrong+=1
                
            elif not el_gt and not int(-1) in el_p: #! -1 in el_gt
                #print(" +++++++ {} early, {}, {}".format(index, el_gt, el_p))
                if early > valutearly[file]:
                    valutearly[file]=early
                early+=1
                #print(index, rowsize-2)
        
        elif state == "a-rec": #* in semi-rec status
            print("[A-REC] in A-REC at frame number {}".format(index))

            for k, v in possible_transitions.items():
                number_of_activated_cameras[index]+=1
                #print(k, v)
                camera=int(k)
                actaivated_camera_indexs[index].append(camera)
                
                processed_frames[camera] += 1
                value = row['Camera {}'.format(camera)]
                rec_counter[camera] -=1

                if '-1' not in value:
                    print("[A-REC] found target in camera {} in {}th frame".format(camera, index))
                    cnt_seen_target[camera]+=1
                    
                    recording[camera].append({
                        "start": index,
                        "duration": 0,
                        "tracks": [get_point(value)],
                        "end": -1,
                    })
                    state="trk"
                    cam_tracking=camera
                    perform_handover = {0: (-1, -1), 1: (-1,-1)}
                    possible_transitions = {} #* clear possible transition history.
                    cnt_camera_choice[index]=[camera]

            if rec_counter[camera] < 0 : #! waits couple frames until saying its gone.
                print("[A-REC] going to REC")

                cam_tracking = -1
                possible_transitions = {}
                state="rec"
                trktimer=0
                


    #print("late {}".format(valutmissing))
    #print("wrong {}".format(valutwrong))               
    #print("early {}".format(valutearly))
    #? calculate results for each iteration. 
    for k, v in processed_frames.items():
        sump += v
    for k, v in cnt_seen_target.items():
        sumst += v
    
    print("cnt_rec: {}, cnt_trk: {}, cnt_trans: {}. total frames: {}. carla frames: {}".format(cnt_rec, cnt_trk, cnt_trans, cnt_total, cnt_target))
    print("percentage of frames processed / total number of frames {}".format(100*sump/(cnt_total*10))) #* processed frames from the vid
    print("percentage of target / total number of frames {}".format(100*cnt_target/(cnt_total*10))) #* target frames in the video
    print("precision {}".format(100*sumst/sump)) #* precison 
    
    if cnt_target ==0:
        print("accuracy {}".format(0)) #* accuracy
        results_for_excel = results_for_excel.append(pd.Series(data=[file, 100*sump/(cnt_total*10), 100*cnt_target/(cnt_total*10), 100*sumst/sump, 0], index=results_for_excel.columns, name=name))
    else:
        print("accuracy {}".format(100*sumst/cnt_target)) #* accuracy
        results_for_excel = results_for_excel.append(pd.Series(data=[file, 100*sump/(cnt_total*10), 100*cnt_target/(cnt_total*10), 100*sumst/sump, 100*sumst/cnt_target], index=results_for_excel.columns, name=name))

    #camera_evaluation_results = camera_evaluation_results.append(pd.Series)
    camera_evaluation_results['gt_'+str(name)] = pd.Series(cnt_camera_ans)
    camera_evaluation_results['prop_'+str(name)] = pd.Series(cnt_camera_choice)
    activation_results[file[:-4]] = pd.Series(actaivated_camera_indexs)

    file_end_time = time.time()
    name+=1

for k, v in valutmissing.items():
    summ +=v
print("[INFO] Average early frames {}".format(summ / (int(file_count/skip_file)+2)))
sim_end_time = time.time()
print("[INFO] total time {}".format(sim_end_time - start_time))        
######! logfiles        

early_late_results["late"] = pd.Series(valutmissing)
early_late_results["wrong"] = pd.Series(valutwrong)
early_late_results["early"] = pd.Series(valutearly)

with pd.ExcelWriter("results/scenario_results_newsimdata_prop_testrun.xlsx", mode='a') as writer:
   results_for_excel.to_excel(writer, sheet_name=shname)
with pd.ExcelWriter("activation_graph/prop_newsimdata_activation_testrun.xlsx", mode='a') as writer:
    activation_results.to_excel(writer, sheet_name=shname)

#* to test if camera selection is wrong or camera time transition is wrong.
# with pd.ExcelWriter("evaluate_prop_dir_or_time.xlsx", mode='a') as writer:
#     camera_evaluation_results.to_excel(writer, sheet_name=shname)

# with pd.ExcelWriter("early_late_evaluation.xlsx", mode='a') as writer:
#     early_late_results.to_excel(writer, sheet_name=shname)

print("[INFO] DONE! ")