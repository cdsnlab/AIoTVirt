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
        return stats.mode(remaining_durations)[0][0], stats.mode(remaining_transitions)[0][0]
        # return int(statistics.median(remaining_durations)), int(statistics.median(remaining_transitions))
    return -1, 0


def preprocess_track(track, vl):
    # if len(track) < vl:
    #     continue
    # TODO NORMALIZE THIS FUCKING TRACK
    output = []
    btw = math.floor(len(track) / float(vl))
    for x, y in track[::btw]:
        if len(output) == vl:
            break
        else:
            output.append((x / 1280, y / 720))
    return np.array([output])



# threshold = 30
# models = {cam: load_model('model_100epochs/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
# passed_frames = 0


def estimate_transition():
    global perform_handover, transition, models, cam_tracking, shit_predictions, passed_frames
    # transition[camera][-1]['tracks'].append(get_point(value))
    if recording[camera][-1]['duration'] > threshold:
        tracks = recording[camera][-1]['tracks']
        full_track = np.array(tracks)
        len_tracks = len(tracks)
        track = preprocess_track(tracks, 30)
        predicted_camera = models[camera].predict(track)
        if shit_predictions > -2:
            predicted_camera = np.argmax(predicted_camera)
        else:
            p = predicted_camera[0].argsort()  # [-2:][0]
            predicted_camera = p[-2:][::-1][1]
        if passed_frames % 4 != 0:
            return
        passed_frames += 1
        handover_time, transition_duration = estimate_handover(camera, predicted_camera, full_track, len_tracks, 10, True, True)

        # TODO Adjust handover time based on threshold and container_boot_time
        if handover_time != -1:
            shit_predictions = 0
            perform_handover = (
                handover_time + transition_duration + index - container_boot_time - recording[camera][-1]['duration'],
                predicted_camera)
            # * As we are tracking in this camera, we don't need to check the rest
            cam_tracking = camera
        else:
            shit_predictions += 1


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
filenames = os.listdir(path)
filenames = filenames[::11]
# filenames = filenames[7:]
MIN_LENGTH = 15
results = {}
transitions = {}
possible_transitions = {} # possible camera and time.
rec_counter ={}
#possible_transitions = {i: -1 for i in range(10)}
container_boot_time = 15  # container boot time
tracker = 0
start_time = time.time()
transition_times = get_transition_dist(path,False) # tmap

load_tmap_time = time.time()
print("loaded tmap in {} seconds".format(load_tmap_time-start_time))
time.sleep(2)
for file in tqdm(filenames):
    if '.csv' not in file:
        continue
    print(file)
    data = pd.read_csv(path + file)

    processed_frames = {i: 0 for i in range(10)}
    cam_tracking = -1
    last_camera = -1
    shit_predictions = 0
    recording = {i: [] for i in range(10)}

    perform_handover = (-1, -1)  # * Frame index; Camera
    # data = pd.read_csv(args.path)
    gr_truth = get_correct(data)
    recovery = True
    handover_points = []
    available_states = ["rec", "trk", "trans"]
    state = "rec" # inital state
    #cnt_rec, cnt_nf, cnt_trk, cnt_trans, cnt_total = 0,0,0,0,0 # count how many each frame was in each state.
    cnt_ = {key: 0 for key in available_states}
    cnt_total = 0
    for index, row in data.iterrows(): #* read frame by frame.
        # * Spencers approach * #
        cnt_[state]+=1
        cnt_total+=1

        if state=="rec": #* recovery: turn on all cameras to search for the target.
            print("[INFO] in REC at frame number {}".format(index))
            for camera in range(10):
                value = row['Camera {}'.format(camera)]
                processed_frames[camera] += 1
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
                    continue

        elif state == "trk": #* tracking: 
            processed_frames[cam_tracking] +=1
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
                for i in transition_map[camera]: # get neighboring cameras transition diff.
                    print("[INFO] possible transition from {} to {} in {} frames".format(camera, i, transition_times[str(camera)+"-->"+str(i)]))
                    possible_transitions[i] = int(transition_times[str(camera)+"-->"+str(i)])-container_boot_time
                    rec_counter[i]=0
                continue
            else: # * If person is here, continue recording trace
                state="trk"
                recording[camera][-1]['tracks'].append(get_point(value))
                

        elif state == "trans": #* in between cams
            print("[INFO] in TRANS at frame number {}".format(index))
            # 1) look at possible transition points and deduct 1 every iteration, 
            # 2) if it hits 0, look at the column values in it.
            # 3) if values are -1 wait for the last one in possible transition
            if not possible_transitions:
                state = "rec"
                break
            for k, v in possible_transitions.items():
                print("[INFO] possible transition point from camera {} to camera {} in {}th frame, {} more to go".format(prev_tracking, k, index, v))
                if v <= 0: # transition time is up for k camera. look at the columns in it                    
                    camera=k
                    processed_frames[camera] += 1
                    value = row['Camera {}'.format(camera)]

                    if '-1' not in value:
                        # * correct transfer.
                        print("[INFO] found target in camera {} in {}th frame".format(camera, index))
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
                    rec_counter[k] += 1
                    if rec_counter[k] >= 15:
                        print("[INFO] pop transition destination for {} at {}th frame".format(k, index))
                        possible_transitions.pop(k)
                        break

                
                # deduct 1 every iteration
                if k in possible_transitions:
                    possible_transitions[k] -=1
            
               

        
        
        
    ####
    '''
    if recovery:
        # * Bruteforce for starting position
        for camera in range(10):
            value = row['Camera {}'.format(camera)]
            processed_frames[camera] += 1
            if cam_tracking == -1 or cam_tracking == camera:
                if '-1' not in value:
                    # * This is a new trace, initialise it
                    recording[camera].append({
                        "start": index,
                        "duration": 0,
                        "tracks": [get_point(value)],
                        "end": -1,
                        "recovery": recovery
                    })
                    cam_tracking = camera
                    recovery = False
                    continue
    elif perform_handover[0] == index:
        # handover_points.append((index, cam_tracking, perform_handover[1]))
        if cam_tracking != -1:
            print(111)
            processed_frames[cam_tracking] += 1
            recording[cam_tracking][-1]['duration'] += 1
            recording[cam_tracking][-1]['end'] = index

        # last_camera = cam_tracking
        cam_tracking = perform_handover[1]
        value = row['Camera {}'.format(cam_tracking)]
        recording[cam_tracking].append({
            "start": index,
            "duration": 0,
            "tracks": [get_point(value)],
            "end": -1,
            "recovery": recovery
        })
        perform_handover = (-1, -1)
    elif cam_tracking != -1:
        processed_frames[cam_tracking] += 1
        # * We have a camera that is tracking, continue here
        camera = cam_tracking
        value = row['Camera {}'.format(camera)]
        # * Increase the current trace's duration
        recording[camera][-1]['duration'] += 1
        # * If person not here, end current trace
        if '-1' in value:
            # handover_points.append((index, cam_tracking, perform_handover[1]))
            recording[cam_tracking][-1]['end'] = index
            cam_tracking = -1
            if perform_handover[0] == -1 or perform_handover[0] < index:
                recovery = True
            else:
                cam_tracking = -1

            # ? Do we have to quickly switch to another camera in this case?
        else:
            recording[camera][-1]['tracks'].append(get_point(value))
            estimate_transition()
    else:
        # TODO Handle wrong predictions where we stay in the same camera

        if last_camera != -1:
            value = row['Camera {}'.format(last_camera)]
            if '-1' not in value:
                camera = last_camera
                estimate_transition()
                # * We continue last trace
                cam_tracking = last_camera
                recording[cam_tracking][-1]['duration'] += 1
                recording[cam_tracking][-1]['tracks'].append(get_point(value))
                recording[cam_tracking][-1]['end'] = -1
                continue
        for camera in range(10):
            value = row['Camera {}'.format(camera)]
            processed_frames[camera] += 1
            if cam_tracking == -1 or cam_tracking == camera:
                if '-1' not in value:
                    if len(recording[camera]) != 0 and recording[camera][-1]['end'] == -1:
                        # * If this is a continuation of a trace, append to it
                        recording[camera][-1]['duration'] += 1
                        estimate_transition()
                    else:
                        # * If this is a new trace, initialise it
                        recording[camera].append({
                            "start": index,
                            "duration": 0,
                            "tracks": [get_point(value)],
                            "end": -1,
                            "recovery": recovery
                        })
                        recovery = False
                        cam_tracking = camera
                        continue
        '''
            ### 

    # print(processed_frames)
    # print(handover_points)
    # results[file.replace('.csv', '')] = {
    #     'predicted': handover_points,
    #     'actual': gr_truth
    # }
    #transitions[file.replace('.csv', '')] = recording
    #tracker += 1
    #passed_frames = 0
    # slacknoti("{} done at {}, {} out of {} remaining".format(file, time.strftime("%H:%M:%S", time.localtime()), tracker, len(filenames)))

# with open("results/transitions_LeNC.json", "w") as file:
#     # print(transitions)
#     file.write(json.dumps(transitions))
    time.sleep(1)
