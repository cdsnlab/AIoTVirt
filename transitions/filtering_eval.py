import filtering
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
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
for camera in range(10):
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

with pd.ExcelWriter("results_full.xlsx") as writer:
    duration_results.to_excel(writer, sheet_name="Duration")
    transition_results.to_excel(writer, sheet_name="Transition")
    combined_results.to_excel(writer, sheet_name="Combined")

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
