
'''
this program evaluates the transition time of LENC (boyan)
'''

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
from termcolor import cprint
import requests

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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



container_boot_time = 0  # ! What do we do with this?
threshold = 30
models = {cam: load_model('model_100epochs/cam_{}.h5'.format(cam), compile=False) for cam in range(10)}
passed_frames = 0


def estimate_transition():
    global perform_handover, transition, models, cam_tracking, shit_predictions, passed_frames
    # transition[camera][-1]['tracks'].append(get_point(value))
    if transition[camera][-1]['duration'] > threshold:
        tracks = transition[camera][-1]['tracks']
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
                handover_time + transition_duration + index - container_boot_time - transition[camera][-1]['duration'],
                predicted_camera)
            # * As we are tracking in this camera, we don't need to check the rest
            cam_tracking = camera
        else:
            shit_predictions += 1


def get_sequences(data: pd.DataFrame):
    seq_traces = []

    labels = []
    cameras = {}
    for i in range(10):
        camera = []
        current_sequence = 0
        start = 0
        labels.append('Camera {}'.format(i))
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

filenames = os.listdir('/home/spencer/samplevideo/multi10zone_allcsv/')
filenames = filenames[::11]
# filenames = filenames[7:]
results = {}
transitions = {}
tracker = 0
for file in tqdm(filenames):
    if '.csv' not in file:
        continue
    data = pd.read_csv("/home/spencer/samplevideo/multi10zone_allcsv/" + file)

    processed_frames = {i: 0 for i in range(10)}
    cam_tracking = -1
    last_camera = -1
    shit_predictions = 0
    transition = {i: [] for i in range(10)}
    # transition_times = {} # TODO GET FROM SIYOUNG
    perform_handover = (-1, -1)  # * Frame index; Camera
    # data = pd.read_csv(args.path)
    gr_truth = get_correct(data)
    recovery = True
    handover_points = []
    for index, row in data.iterrows():
        if recovery:
            # * Bruteforce for starting position
            for camera in range(10):
                value = row['Camera {}'.format(camera)]
                processed_frames[camera] += 1
                if cam_tracking == -1 or cam_tracking == camera:
                    if '-1' not in value:
                        # * This is a new trace, initialise it
                        transition[camera].append({
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
                transition[cam_tracking][-1]['duration'] += 1
                transition[cam_tracking][-1]['end'] = index

            # last_camera = cam_tracking
            cam_tracking = perform_handover[1]
            value = row['Camera {}'.format(cam_tracking)]
            transition[cam_tracking].append({
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
            transition[camera][-1]['duration'] += 1
            # * If person not here, end current trace
            if '-1' in value:
                # handover_points.append((index, cam_tracking, perform_handover[1]))
                transition[cam_tracking][-1]['end'] = index
                cam_tracking = -1
                if perform_handover[0] == -1 or perform_handover[0] < index:
                    recovery = True
                else:
                    cam_tracking = -1

                # ? Do we have to quickly switch to another camera in this case?
            else:
                transition[camera][-1]['tracks'].append(get_point(value))
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
                    transition[cam_tracking][-1]['duration'] += 1
                    transition[cam_tracking][-1]['tracks'].append(get_point(value))
                    transition[cam_tracking][-1]['end'] = -1
                    continue
            for camera in range(10):
                value = row['Camera {}'.format(camera)]
                processed_frames[camera] += 1
                if cam_tracking == -1 or cam_tracking == camera:
                    if '-1' not in value:
                        if len(transition[camera]) != 0 and transition[camera][-1]['end'] == -1:
                            # * If this is a continuation of a trace, append to it
                            transition[camera][-1]['duration'] += 1
                            estimate_transition()
                        else:
                            # * If this is a new trace, initialise it
                            transition[camera].append({
                                "start": index,
                                "duration": 0,
                                "tracks": [get_point(value)],
                                "end": -1,
                                "recovery": recovery
                            })
                            recovery = False
                            cam_tracking = camera
                            continue

    # print(processed_frames)
    # print(handover_points)
    # results[file.replace('.csv', '')] = {
    #     'predicted': handover_points,
    #     'actual': gr_truth
    # }
    transitions[file.replace('.csv', '')] = transition
    tracker += 1
    passed_frames = 0
    # slacknoti("{} done at {}, {} out of {} remaining".format(file, time.strftime("%H:%M:%S", time.localtime()), tracker, len(filenames)))

with open("results/transitions_LeNC.json", "w") as file:
    # print(transitions)
    file.write(json.dumps(transitions))
