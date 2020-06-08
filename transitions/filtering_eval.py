import filtering
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_archive', default='transitions/traces_train_test.npz', 
                    type=str, help='Path to NPZ file')
parser.add_argument('-o', '--output_file', default='filtering_results', type=str, help='File where the image results are stored')
args = parser.parse_args()

def rmse(y_true, y_pred): return np.sqrt(mse(y_true, y_pred))


data = np.load(args.input_archive, allow_pickle=True)

duration_results = pd.DataFrame(
    columns=["source_cam", "target_cam", "trace_%", "angle_lim", "count", "mse", "rmse", "mae"])
transition_results = pd.DataFrame(
    columns=["source_cam", "target_cam", "trace_%", "angle_lim", "count", "mse", "rmse", "mae"])
combined_results = pd.DataFrame(
    columns=["source_cam", "target_cam", "trace_%", "angle_lim", "count", "mse", "rmse", "mae"])

trace_perc = 0.8
angle_lim = 10
name = 0

for camera in range(10):
    cam_data = data['cam_{}'.format(camera)]
    train = cam_data[0]
    test = cam_data[1]
    true_duration = []
    true_transition = []
    true_combined = []
    predicted_duration = []
    predicted_transition = []
    predicted_combined = []
    for path, target, transition, duration in test:
        partition = int(len(path) * trace_perc)
        trace = path[:partition]
        est_duration, est_transition = filtering.estimate_handover(
            target, trace, len(trace), train, angle_lim=angle_lim)
        true_duration.append(len(path))
        true_transition.append(transition)
        true_combined.append(len(path) + transition)
        predicted_duration.append(est_duration)
        predicted_transition.append(est_transition)
        predicted_combined.append(
            est_duration + est_transition)
        
    if len(true_duration) > 0 and len(true_transition) > 0 and len(true_combined) > 0:
        duration_results = duration_results.append(pd.Series(data=[camera, target, trace_perc, angle_lim, len(true_duration), mse(
            true_duration, predicted_duration), rmse(true_duration, predicted_duration), mae(true_duration, predicted_duration)], index=duration_results.columns, name=name))
        transition_results = transition_results.append(pd.Series(data=[camera, target, trace_perc, angle_lim, len(true_duration), mse(
            true_transition, predicted_transition), rmse(true_transition, predicted_transition), mae(true_transition, predicted_transition)], index=duration_results.columns, name=name))
        combined_results = combined_results.append(pd.Series(data=[camera, target, trace_perc, angle_lim,  len(true_duration), mse(
            true_combined, predicted_combined), rmse(true_combined, predicted_combined), mae(true_combined, predicted_combined)], index=duration_results.columns, name=name))
        name += 1

with pd.ExcelWriter("transitions/{}.xlsx".format(args.output_file)) as writer:
    duration_results.to_excel(writer, sheet_name="Duration")
    transition_results.to_excel(writer, sheet_name="Transition")
    combined_results.to_excel(writer, sheet_name="Combined")
