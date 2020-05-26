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
    0: [1,8,9],
    1: [0, 8, 2],
    2: [1, 3, 7],
    3: [1, 2, 4],
    4: [3, 5],
    5: [4, 6],
    6: [5, 7],
    7: [6, 3, 8],
    8: [1,7,9],
    9: [0,8]
}



models = {cam: load_model('/home/boyan/LSTM_Cam_pred/model_100epochs/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
#models = {cam: load_model('/home/boyan/LSTM_Cam_pred/model_new_label/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
passed_frames = 0

for camera in range(10):
    loaded = np.load("/home/boyan/out_label_trans/{}.npy".format(camera), allow_pickle=True)
    #loaded = np.load("/home/boyan/out_label_neighb_irw/{}.npy".format(camera), allow_pickle=True)
    # loaded = np.load("/home/boyan/out_label_neighb/{}.npy".format(camera), allow_pickle=True)
    traces[camera] = loaded

threshold = 30


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
    point = strpoint.replace('(', '').replace(')', '').split(',')
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
    for path, label, transition in traces[source][:-1]:
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
        return stats.mode(remaining_durations)[0][0], stats.mode(remaining_transitions)[0][0]
        # return int(statistics.median(remaining_durations)), int(statistics.median(remaining_transitions))
    else:
        #print("[DEBUG] Nothing for {} -> {} ".format(source, dest))
        return -1, 0


def preprocess_track(track, vl):
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
            output.append((x / 1280, y / 720))
    return np.array([output])




def estimate_transition():
    global perform_handover, transition, models, cam_tracking, passed_frames
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
                if predicted_camera[0][p[-2:][::-1][i]]>0.3: #! this is cuz all classes must add up to 1. 
                    #print("[DEBUG] added {} as {} option".format(p[-2:][::-1][i], i ))
                    choice.append(p[-2:][::-1][i])
                else:
                    choice.append(-1)
     

            for i, target in enumerate (choice):
                #* handover_t: how long the left trace is.., transition_d: how long it took from camera a to b
                handover_t, transition_d = estimate_handover(camera, target, full_track, len_tracks, 10, False, False)
                #print(target, handover_t, transition_d)
                handover_time.append(handover_t)
                transition_duration.append(transition_d)
                #print("[DEBUG] destination {}, handover_time {}, transition_dur {}".format(k, handover_t, transition_d)) 
                # TODO Adjust handover time based on threshold and container_boot_time
                if handover_t!=-1:
                    perform_h = (handover_t + transition_d + index - container_boot_time, target) # - recording[camera][-1]['duration'], target)
                    perform_handover[i]=(perform_h)

                    # * As we are tracking in this camera, we don't need to check the rest
                    cam_tracking = camera
    passed_frames += 1



path ="/home/spencer1/samplevideo/sim_csv/prev_full/"

path1, dirs1, files1 = next(os.walk(path))
file_count = len(files1)
skip_file=4
#shname="even"
shname="odd4-fixing-0.3-"
filenames = os.listdir(path)
filenames = filenames[1::skip_file]
#filenames = filenames[::skip_file]
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



for file in tqdm(filenames):
    if '.csv' not in file:
        continue
    print(file)
    file_start_time=time.time()
    data = pd.read_csv(path + file)
    rowsize = len(data['Camera 1'])


    
    cam_tracking = -1
    perform_handover = {0: (-1, -1), 1: (-1,-1)}  # * Frame index; Camera
    # data = pd.read_csv(args.path)
    number_of_activated_cameras = {i: 0 for i in range(MAX_SIM)}
    recording = {i: [] for i in range(10)}
    processed_frames = {i: 0 for i in range(10)}
    cnt_seen_target= {i: 0 for i in range(10)}
    cnt_camera_choice ={}
    cnt_camera_ans ={}
    available_states = ["rec", "trk", "trans"]
    state = "rec" # inital state
    cnt_ = {key: 0 for key in available_states}
    cnt_total= 0 #* total number of frames.
    cnt_target=0 #* howmany valid frames there are in this file
    cnt_rec=0 #* how long you were in REC state
    cnt_trk=0 #* how long you were in TRK state
    cnt_trans=0 #* how long you were in TRANS state
    sump=0
    sumst=0
    for index, row in data.iterrows(): #* read frame by frame.
        # * Spencers approach * #
        cnt_[state]+=1
        cnt_total+=1

        if index==rowsize-1:
            for i in range(rowsize-1, MAX_SIM):
                number_of_activated_cameras[i] = -1
        
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
                value = row['Camera {}'.format(camera)]
                processed_frames[camera] += 1
                number_of_activated_cameras[index]+=1
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
                    break

        elif state == "trk": #* tracking: 
            cnt_trk+=1
            processed_frames[cam_tracking] +=1
            camera = cam_tracking
            number_of_activated_cameras[index]+=1
            value = row['Camera {}'.format(camera)]

            if '-1' in value: # * If person not here, end current trace
                print("[INFO] STOP TRK in camera {} frame number {}".format(camera, index))
                recording[cam_tracking][-1]['end'] = index
                prev_tracking = cam_tracking
                cnt_camera_choice[index] = [-1]
                #cam_tracking = -1
                state="trans" #! check if there is any handover points in the future.
                continue
            else: # * If person is here, continue recording trace
                # * We have a camera that is tracking, continue here
                
                cnt_camera_choice[index] = [cam_tracking]
                cnt_seen_target[camera]+=1
                recording[camera][-1]['tracks'].append(get_point(value))
                print("[INFO] in TRK in camera {} frame number {}".format(camera, index))
 
                # * Increase the current trace's duration
                recording[camera][-1]['duration'] += 1
                            
            estimate_transition() #! check if the index of the prediction matches current index

            for k, v in perform_handover.items(): #! regardless of if the person is there or not, keep updating possible transitions.
                print("[INFO] these are possible transitions {}".format(v))
                if v[0] != -1:
                    possible_transitions[k] = v
                    rec_counter[k] = 0
               

        elif state == "trans": #* in between cams
            cnt_trans+=1
            cnt_camera_choice[index]=[-1]
            print("[INFO] in TRANS at frame number {}".format(index))
            if not possible_transitions:
                print("[DEBUG] going back to REC")
                state = "rec"
                continue
            for k, v in possible_transitions.items():
                if v[0]<=index: #transition time == index
                    print("[DEBUG] possible transition point from camera {} to camera {} at {}th frame".format(prev_tracking, v[1], v[0]))

                    camera=v[1]
                    processed_frames[camera] += 1
                    value = row['Camera {}'.format(camera)]
                    if '-1' not in value:
                        # * correct transfer.
                        print("[INFO] found target in camera {} in {}th frame".format(camera, index))
                        cnt_seen_target[camera]+=1
                        number_of_activated_cameras[index]+=1
                        recording[camera].append({
                            "start": index,
                            "duration": 0,
                            "tracks": [get_point(value)],
                            "end": -1,
                        })
                        state="trk"
                        cam_tracking=camera
                        perform_handover = {0: (-1, -1), 1: (-1,-1)}
                        cnt_camera_choice[index]=[camera]
                        break
                    rec_counter[k]+=1
                else: #! what if they are far ahead? do nothing until it reaches that number.
                    print("[DEBUG] waiting until {}th frame".format(v[0]))

                #elif v[0] < index: 
                if rec_counter[k] >= 30: #* give 30 frames until calling missing.
                    print("[INFO] pop transition destination for {} at {}th frame".format(k, index))
                    possible_transitions.pop(k)
                    
                    break
        

                            
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
    activation_results[file[:-4]+"_activenumber"] = pd.Series(number_of_activated_cameras)

    file_end_time = time.time()
    name+=1
sim_end_time = time.time()
print("[INFO] total time {}".format(sim_end_time - start_time))        
######! logfiles        
with pd.ExcelWriter("results/scenario_results_prop.xlsx", mode='a') as writer:
   results_for_excel.to_excel(writer, sheet_name=shname)

#* to test if camera selection is wrong or camera time transition is wrong.
with pd.ExcelWriter("evaluate_prop_dir_or_time.xlsx", mode='a') as writer:
    camera_evaluation_results.to_excel(writer, sheet_name=shname)

with pd.ExcelWriter("activation_graph/prop_activation.xlsx", mode='a') as writer:
    activation_results.to_excel(writer, sheet_name=shname)
print("[INFO] DONE! ")