import filtering
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import statistics

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import mean_absolute_percentage_error as mae


def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))


data = np.load('out.npz', allow_pickle=True)
data = data['arr_0']

duration_results = pd.DataFrame(
    columns=["source_cam", "target_cam", "trace_%", "angle_lim", "count", "mse", "rmse", "mae"])
transition_results = pd.DataFrame(
    columns=["source_cam", "target_cam", "trace_%", "angle_lim", "count", "mse", "rmse", "mae"])
combined_results = pd.DataFrame(
    columns=["source_cam", "target_cam", "trace_%", "angle_lim", "count", "mse", "rmse", "mae"])

trace_perc = 0.9
trace_percentages = [0.75, 0.8, 0.85, 0.9]
angle_lim = 5
angle_lims = [5,10,15,20]
name = 0
for camera in tqdm(range(10)):
    cam_data = data[camera]
    for target in range(10):
        if target == camera:
            continue
        fig = make_subplots(rows=1, cols=2)
        train = cam_data[target - 1][0]
        test = cam_data[target - 1][1]

        true_duration = []
        true_transition = []
        true_combined = []
        predicted_duration = []
        predicted_transition = []
        predicted_combined = []
        for trace_perc in trace_percentages:
            for angle_lim in angle_lims:
                for path, label, transition in test:
                    if label == target:
                        partition = int(len(path) * trace_perc)
                        trace = path[:partition]
                        est_duration, est_transition = filtering.estimate_handover(
                            target, trace, len(trace), train, angle_lim=angle_lim)
                        true_duration.append(len(path))
                        true_transition.append(transition)
                        true_combined.append(len(path) + transition)
                        predicted_duration.append(est_duration)
                        predicted_transition.append(est_transition)
                        predicted_combined.append(est_duration + est_transition)

                if len(true_duration) > 0 and len(true_transition) > 0 and len(true_combined) > 0:
                    duration_results = duration_results.append(pd.Series(data=[camera, target, trace_perc, angle_lim, len(true_duration), mse(
                        true_duration, predicted_duration), rmse(true_duration, predicted_duration), mae(true_duration, predicted_duration)], index=duration_results.columns, name=name))
                    transition_results = transition_results.append(pd.Series(data=[camera, target, trace_perc, angle_lim, len(true_duration), mse(
                        true_transition, predicted_transition), rmse(true_transition, predicted_transition), mae(true_transition, predicted_transition)], index=duration_results.columns, name=name))
                    combined_results = combined_results.append(pd.Series(data=[camera, target, trace_perc, angle_lim,  len(true_duration), mse(
                        true_combined, predicted_combined), rmse(true_combined, predicted_combined), mae(true_combined, predicted_combined)], index=duration_results.columns, name=name))
                name += 1
        # plot_duration = go.Histogram(x=results_duration, nbinsx=len(results_duration))
        # plot_transition = go.Histogram(x=results_transition, nbinsx=len(results_transition))
        # fig.append_trace(plot_duration, 1, 1)
        # fig.append_trace(plot_transition, 1, 2)
        # fig.write_html("results/cam_{}_target_{}.html".format(camera, target))
        # del fig

with pd.ExcelWriter("compare_transition_time.xlsx") as writer:
    duration_results.to_excel(writer, sheet_name="Duration")
    transition_results.to_excel(writer, sheet_name="Transition")
    combined_results.to_excel(writer, sheet_name="Combined")

###
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

    transition_results = pd.DataFrame(columns=["source_cam", "target_cam", "mse", "rmse", "mae"])
    name = 0
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
                predy.append(gettransitiontime(tmap, str(traces[i]["camera"])+"-->"+str(trace["camera"])))
                acty.append(trace['start'] - traces[i]['end'])
                transition_results = transition_results.append(pd.Series(data=[str(traces[i]["camera"]), str(trace["camera"]), mse(acty, predy), rmse(acty, predy), mae(acty, predy)], index=transition_results.columns, name=name))
            name+=1
    with pd.ExcelWriter("compare_transition_time.csv", mode='a') as writer:
        transition_results.to_excel(writer, sheet_name="previous approach")
    #return traces, cameras, labels, transition

def gettransitiontime(tmap, direction):
    for k in (tmap):
        if k == direction:
            return tmap[k]
    return -1


get_transition_dist('/home/spencer1/samplevideo/sim_csv/start1_endall_csv', False)




# target_cam = 9

# # for trace in test[:30]:
# results = {}
# i = 0
# for target_cam in [1]:
#     results[target_cam] = {
#         "duration": [],
#         "transition": []
#     }
#     for path, label, transition in tqdm(test):
#         if label == target_cam:
#             partition = int(len(path) * 0.90)
#             trace = path[:partition]
#             est_duration, est_transition = filtering.estimate_handover(target_cam, trace, len(trace), train, angle_lim=5)
#             # print(abs(len(path) - duration), abs(est_transition - transition))
#             results[target_cam]["duration"].append(len(path) - est_duration)
#             # print(est_transition)
#             results[target_cam]["transition"].append(transition - est_transition)
#             i += 1

# fig = go.Figure(data=[go.Histogram(x=results[1]['duration'], nbinsx=len(results[1]['duration']))])
# fig.write_html("duration.html")


# fig.write_image("duration.pdf")
# print(results[9]['duration'])
# plt.hist(results[9]['duration'], bins=len(results[9]['duration']))
# plt.savefig("duration.png")
# plt.clf()
# print(results[9]['transition'])
# plt.hist(results[9]['transition'], bins=len(results[9]['transition']))
# plt.savefig("transition.png")
