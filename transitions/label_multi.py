import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple
import numpy as np
import os
import time, math

class Trace(NamedTuple):
    start: int
    end: int
    duration: int
    camera: int


def get_naive(data: pd.DataFrame):
    traces = []
    current_cam = -1
    start = 0
    end = 0
    current_sequence = 0
    trace_found = False

    c = 0
    for index, row in data.iterrows():
        seen_trace = False
        for cam, value in enumerate(row.values[1:]):
            # print(cam, value)
            if value != '(-1, -1)':
                seen_trace = True
                if current_cam == -1:
                    current_cam = cam
                    start = index
                    current_sequence += 1
                    break
                elif current_cam == cam:
                    found = False
                    for other, val in enumerate(row.values[1+cam:]):
                        if val != '(-1, -1)':
                            traces.append(Trace(start, index, current_sequence, current_cam))
                            current_cam = other
                            start = index
                            current_sequence = 0
                            found = True
                            break
                    if not found:
                        current_sequence += 1
                    break
                else:
                    # print(cam)
                    traces.append(Trace(start, index, current_sequence, current_cam))
                    current_cam = cam
                    start = index
                    current_sequence = 0
                    break
        if not seen_trace:
            if current_cam != -1:
                traces.append(Trace(start, index, current_sequence, current_cam))
                current_sequence = 0
            current_cam = -1
            seen_trace = False
    if current_sequence != 0:
        traces.append(Trace(start, index, current_sequence, current_cam))
    print(traces)




def get_sequences(data: pd.DataFrame):
    traces = []

    labels = []
    cameras = {}
    for i in range(10):
        camera = []
        current_sequence = 0
        start = 0
        labels.append('Camera {}'.format(i))
        for index, value in data['Camera {}'.format(i)].items():
            if '-1' not in value:
                if current_sequence == 0:
                    start = index
                current_sequence += 1
            else:
                if current_sequence != 0 and current_sequence > 15:
                    camera.append((start, current_sequence))
                    traces.append(Trace(start, index, current_sequence, i))
                current_sequence = 0
        if current_sequence != 0 and current_sequence > 15:
            camera.append((start, current_sequence))
            traces.append(Trace(start, index, current_sequence, i))
        cameras[i] = camera
    return traces, cameras, labels

def get_cam_order(traces):
    for i, trace in enumerate(traces):
        for other in traces[i:]:
            if other.start > trace.start and other.end <= trace.end:
                # print(other)
                traces.remove(other)
    i = 1
    to_remove = []
    for trace in traces[1:-1]:
        try:
            if trace.start < traces[i-1].end and trace.end > traces[i+1].start and traces[i-1].end > traces[i+1].start:
                traces.remove(trace)
                i -= 1
            i += 1
        except IndexError:
            print(trace)
            pass
    return traces

def get_simple(traces):
    final = []
    try:
        final.append(traces[0])
        for i, trace in enumerate(traces[1:]):
            if trace.start < final[-1].end and trace.end > final[-1].end:
                final.append(Trace(final[-1].end, trace.end, trace.end - final[-1].end, trace.camera))
            elif trace.start > final[-1].end:
                final.append(trace)
    except IndexError:
        pass
    return final

def get_neighbour(traces):
    global transition_map
    final = []
    count = 0
    try:
        final.append(traces[0])
        for i, trace in enumerate(traces[1:-1]):
            src = final[-1].camera
            transitions = []
            for t in traces[len(final):]:
                if t.camera in transition_map[src]:
                    transitions.append(t)

            for nT in transitions:
                # if nT.camera in transition_map[src]:
                if nT.start > final[-1].start and nT.end < final[-1].end:
                    continue
                if nT.start >= final[-1].end:
                    final.append(nT)
                    break
                else:
                    if final[-1].end == nT.end:
                        a = 0
                        continue
                    elif nT.end < final[-1].end:
                        continue
                    final.append(Trace(final[-1].end, nT.end, nT.end - final[-1].end, nT.camera))
                    break
                # else:
                #     count += 1
            # trans_traces = [ for trace in traces[len(final) -1:]]
    except IndexError:
        pass

    return final


def preprocess_track(series, label, min_end_point=-1, vl=30):
    inputs, outputs = [], []
    portions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    rwset = [100, 90, 80, 70, 60, 50]
    # for series in track:
    for prt in portions:
        startloc = int(len(series) / 100 * prt)
        for rw in rwset:
            endloc = int(len(series) / 100 * rw)
            # * If we are gonna overlap with previous trajectories
            if min_end_point != -1 and endloc < min_end_point:
                break
            btw = math.floor((endloc - startloc) / float(vl))
            tmpx, tmpy = [], []
            cnt = 0
            if (endloc - startloc) < vl:  # if there are less plots than vl
                continue
            for x, y in series[startloc:endloc:btw]:
                if cnt == vl:
                    break
                else:
                    tmpx.append(np.array([x, y]))
                cnt += 1
            # tmpy.append(label)
            inputs.append(np.array(tmpx))
            outputs.append(label)
    return inputs, outputs, len(inputs) == 0


def label_files(path, naive=False, timestamp_thresh=0.8):
    filenames = os.listdir(path)
    train_set = {camera: [] for camera in range(10)}
    test_set = {camera: [] for camera in range(10)}
    count = 0
    good_count = 0
    for file in filenames:
        if '.csv' in file:
            # print(file)
            data = pd.read_csv(path + "/" + file)
            traces, _, _ = get_sequences(data)
            traces = sorted(traces, key=lambda t: t.start)
            # * For every camera from start to end -1 - last one has no transition
            for i, source in enumerate(traces[:-1]):
                # * For every subsequent camera of that one (aka possible transitions)
                # src_camera = source.camera
                has_overlapping = False
                min_end_points = []
                for j, target in enumerate(traces[i:]):
                    if target.start >= source.start and target.end <= source.end:
                        # * If trace is completely contained within ours
                        continue

                    if target.start <= source.end and target.end > source.end:
                        # TODO check if next overlaps with source
                        try:
                            if traces[i+j+1].start <= source.end:
                                # * If trace overlaps ours
                                has_overlapping = True
                                track_start = source.start
                                track_end = target.start
                                full_track = data['Camera {}'.format(source.camera)][track_start:track_end].values
                                full_track = np.array(
                                    [[int(p) for p in point.replace('(', '').replace(')', '').split(',')]
                                     for point in full_track])
                                min_end_point = -1
                                if len(min_end_points) != 0:
                                    min_end_point = min_end_points[-1]
                                inputs, outputs, empty = preprocess_track(full_track, target.camera, min_end_point)
                                # * Add to output file with a couple of points for testing
                                if not empty:
                                    for ind, item in enumerate(inputs[:-2]):
                                        train_set[source.camera].append(np.array([inputs[ind], outputs[ind]]))
                                    for ind, item in enumerate(inputs[-2:]):#, outputs[-2:]:
                                        test_set[source.camera].append(np.array([inputs[ind], outputs[ind]]))
                                # * Add this trace's start point as a boundary for subsequent traces
                                min_end_points.append(target.start - source.start)
                            else:
                                track_start = source.start
                                track_end = source.end
                                full_track = data['Camera {}'.format(source.camera)][track_start:track_end].values
                                full_track = np.array(
                                    [[int(p) for p in point.replace('(', '').replace(')', '').split(',')]
                                     for point in full_track])
                                inputs, outputs, empty = preprocess_track(full_track, target.camera, -1)
                                # * Add to output file with a couple of points for testing
                                if not empty:
                                    for ind, item in enumerate(inputs[:-2]):
                                        train_set[source.camera].append(np.array([inputs[ind], outputs[ind]]))
                                    for ind, item in enumerate(inputs[-2:]):  # , outputs[-2:]:
                                        test_set[source.camera].append(np.array([inputs[ind], outputs[ind]]))
                        except IndexError:
                            pass

                        # ! case when yes, case when no


                    if target.start >= source.end and not has_overlapping:
                        # TODO preprocess full_track of trace
                        track_start = source.start
                        track_end = source.end
                        full_track = data['Camera {}'.format(source.camera)][track_start:track_end].values
                        full_track = np.array([[int(p) for p in point.replace('(', '').replace(')', '').split(',')]
                                               for point in full_track])
                        inputs, outputs, empty = preprocess_track(full_track, target.camera, -1)
                        # * Add to output file with a couple of points for testing
                        if not empty:
                            for ind, item in enumerate(inputs[:-2]):
                                train_set[source.camera].append(np.array([inputs[ind], outputs[ind]]))
                            for ind, item in enumerate(inputs[-2:]):  # , outputs[-2:]:
                                test_set[source.camera].append(np.array([inputs[ind], outputs[ind]]))
                        break
            # print(train_set)
                
                # transition = traces[i+1].start - trace.end
                # seq = data['Camera {}'.format(cam)][trace.start:trace.end].values
                # seq = [[int(p) for p in point.replace('(', '').replace(')','').split(',')] for point in seq]
                # s = np.array(seq)
                # Following TODO is for the numpy file for filtering transition times
                # # TODO expand to 0-100, 10-100, 20-100, etc
                # for perc in [0.05,0.10,0.15,0.20,0.25, 0.3, 0.35, 0.45]:
                #     st = int(len(s) * perc)
                #     # t = s[:]
                #     files[cam].append(np.array([s[st:], label, transition]))
                # files[cam].append(np.array([s, label, transition]))
        # break
    # print(count, good_count)
    for cam, content in train_set.items():
        ncontent = np.array(content)
        # print(cam, ncontent)
        print(cam, len(content))
        # np.save("new_labeling/train_{}".format(cam), ncontent)
    for cam, content in test_set.items():
        ncontent = np.array(content)
        # print(cam, ncontent)
        print(cam, len(content))
        # np.save("new_labeling/test_{}".format(cam), ncontent)
    return train_set

    
def get_all_traces(path):
    filenames = os.listdir(path)
    files = {camera: [] for camera in range(10)}
    all_traces = []
    for file in filenames:
        if '.csv' in file:
            # print(file)
            data = pd.read_csv(path + "/" + file)
            traces, _, _ = get_sequences(data)
            traces = sorted(traces, key=lambda t: t.start)
            traces = get_cam_order(traces)
            all_traces += traces
    return all_traces

start = time.time()
# label_files('multi10zone_npy', False, timestamp_thresh=0.85)
# print(time.time() - start)
        
all_traces = label_files('multi10zone_npy', True)
print(time.time() - start)
# durs = []
# x = []
# i = 0
# for path, label in all_traces[0]:
#     if label == 1:
#         durs.append(len(path))
#         y = math.hypot(path[0][0] - path[-1][0], path[0][1] - path[-1][1]) #/ len(path)
#         x.append(y)
#         i+=1

# plt.scatter(x,durs)
# plt.savefig('durations.png', dpi=350)
# fs = [
#     "multi10zone_npy/start_2_end_9_run_12.csv",
#     "multi10zone_npy/start_2_end_9_run_16.csv",
#     "multi10zone_npy/start_2_end_8_run_97.csv",
#     "multi10zone_npy/start_2_end_9_run_8.csv",
#     "multi10zone_npy/start_2_end_8_run_77.csv",
#     "multi10zone_npy/start_2_end_8_run_87.csv",
#     "multi10zone_npy/start_2_end_8_run_72.csv",
#     "multi10zone_npy/start_2_end_7_run_8.csv",
# ]

# data = pd.read_csv(fs[3])
# # data = pd.read_csv("multi10zone_npy/start_9_end_5_run_48.csv")

# traces, cameras, labels = get_sequences(data)
# traces = sorted(traces, key=lambda t: t.start)
# # print(traces)
# print(get_cam_order(traces))

# fig, ax  = plt.subplots()

# i = 1
# for c in cameras.values():
#     ax.broken_barh(c, (i*10, 8))
#     i += 1

# ax.set_yticks([14, 24, 34, 44, 54, 64, 74, 84, 94, 104])
# ax.set_yticklabels(labels)
# ax.set_xlim(0, 1600)
# ax.grid(True)
# plt.show()
# plt.savefig('overlap_2.png', dpi=500)
