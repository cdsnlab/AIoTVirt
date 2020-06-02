import pandas as pd
import matplotlib.pyplot as plt
from typing import NamedTuple
import numpy as np
import os
import time, math
# print(data.T)

# data.T.to_csv('test.csv')

# data = data.T
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
    # print("Stage 1 complete \n ###")
    # for t in traces:
        # print(t)
    # print("###")
    i = 1
    to_remove = []
    for trace in traces[1:-1]:
        try:
            if trace.start < traces[i-1].end and trace.end > traces[i+1].start and traces[i-1].end > traces[i+1].start:
                # print(trace)
                # print(traces[i-1].end, traces[i-1].camera)
                # print(traces[i+1].start, traces[i+1].camera)
                # print("=====")
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

def label_files(path, naive=False, timestamp_thresh=0.8):
    filenames = os.listdir(path)
    files = {camera: [] for camera in range(10)}
    for file in filenames:
        if '.csv' in file:
            # print(file)
            data = pd.read_csv(path + "/" + file)
            traces, _, _ = get_sequences(data)
            traces = sorted(traces, key=lambda t: t.start)
            if not naive:
                traces = get_cam_order(traces)
            if naive:
                traces = get_simple(traces)
            # print(traces)
            for i, trace in enumerate(traces[:-1]):
                cam = trace.camera
                label = traces[i+1].camera
                # TODO calculate transition time to next trace
                transition = traces[i+1].start - trace.end
                seq = data['Camera {}'.format(cam)][trace.start:trace.end].values
                seq = [[int(p) for p in point.replace('(', '').replace(')','').split(',')] for point in seq]
                s = np.array(seq)
                # print(s, label)
                # if cam == 7 and label == 0:
                #     print(path + "/" + file)
                # files[cam].append(np.array([s, [label, int(timestamp_thresh * len(s))]]))
                files[cam].append(np.array([s, label, transition]))

    for cam, content in files.items():
        ncontent = np.array(content)
        # print(cam, ncontent)
        np.save("out_label_trans/{}".format(cam), ncontent)
    return files

    
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
        
all_traces = label_files('multi10zone_npy', False)

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