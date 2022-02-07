import math
import numpy as np

def sampling(
    method, 
    trajectory, 
    cam_label, 
    time_label, 
    sampling_rate = 1, 
    seq_length = 30
):
    if method == 'sw-o': #Sliding window complete overlap
        if len(trajectory) < 30:
            return -1, -1, -1
        sampled_traj = []
        sampled_cam_label = []
        sampled_time_label = []
        if len(trajectory) > 60:
            trajectory = trajectory[-60:]

        portion_len = int(len(trajectory) * sampling_rate)

        for i, coordinates in enumerate(trajectory):
            if (i + seq_length) > portion_len:
                break
            sampled_traj.append(trajectory[i : (i + seq_length)])
            sampled_cam_label.append(cam_label)
            sampled_time_label.append(time_label)
        return sampled_traj, sampled_cam_label, sampled_time_label

    elif (method == 'ed') or ((method == 'irw') and (sampling_rate != 1)): # Evenly distributed
        if len(trajectory) > 60:
            trajectory = trajectory[-60:]
        portion_len = int(len(trajectory) * sampling_rate)
        tmp_traj = []
        count = 0
        btw = math.floor(portion_len / float(seq_length))
        if portion_len < seq_length:
            return -1, -1, -1
        for coors in trajectory[::btw]:
            if count == seq_length:
                break
            else:
                tmp_traj.append(coors)
            count += 1
        return tmp_traj, cam_label, time_label
                    
    elif (method == 'last') or ((method == 'sw-o') and (sampling_rate != 1)):
        portion_len = int(len(trajectory) * sampling_rate)
        if len(trajectory) < seq_length:
            return -1, -1, -1
        return trajectory[-seq_length:], cam_label, time_label

def sampling_multi_track(
    method, 
    trajectories, 
    next_cam_label, 
    trans_time_label, 
    sampling_rate = 1, 
    seq_length = 30
):

    sampled_trajectories = []
    sampled_cam_labels = []
    sampled_time_labels = []
    for trajectory, cam_label, time_label in zip(trajectories, next_cam_label, trans_time_label):
        sampled_traj, sampled_cam_label, sampled_time_label = sampling(
            method = method,
            trajectory = trajectory,
            cam_label = cam_label,
            time_label = time_label,
            sampling_rate = sampling_rate,
            seq_length = seq_length
        )
        if sampled_traj == -1:
            continue
        if method == 'sw-o': #Sliding window complete overlap
            sampled_trajectories.extend(sampled_traj)
            sampled_cam_labels.extend(sampled_cam_label)
            sampled_time_labels.extend(sampled_time_label)

        elif (method == 'ed') or ((method == 'irw') and (sampling_rate != 1)): # Evenly distributed
            sampled_trajectories.append(sampled_traj)
            sampled_cam_label.append(sampled_cam_label)
            sampled_time_label.append(sampled_time_label)
                        
        elif (method == 'last') or ((method == 'sw-o') and (sampling_rate != 1)):
            sampled_trajectories.append(sampled_traj)
            sampled_cam_label.append(sampled_cam_label)
            sampled_time_label.append(sampled_time_label)

    return np.array(sampled_trajectories), np.array(sampled_cam_label), np.array(sampled_time_label, dtype = np.float32)
        