import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_dir = 'data'

raw_output_dir = 'data/output_tracks'
extracted_tracks_dir = 'data/extracted_tracks'
num_of_camera = 18

transition_period = 600 #frames
out_of_view_accept_period = 30 #frames

def get_cam_entry_list(df, index, num_of_cam):
    cam_entry_list = {}
    for cam_num in range(num_of_cam):
        cam_id = 'Camera ' + str(cam_num)
        current_entry = df.iloc[index][cam_id]
        current_entry = current_entry.replace('(', '')
        current_entry = current_entry.replace(')', '')
        if current_entry != '-1, -1':
            temp = current_entry.split(', ')
            frame, coor_x, coor_y = (int(x) for x in temp[0:3])
            cam_entry_list[cam_num] = (frame, coor_x, coor_y)
        else:
            cam_entry_list[cam_num] = (-1, -1)
    return cam_entry_list

trajectory_list = {}
for cam_num in range(num_of_camera):
    trajectory_list[cam_num] = []

for file in tqdm(os.listdir(raw_output_dir)):
    print(file)
    file_path = os.path.join(raw_output_dir, file)
    #file_path = '/home/tung/carlasim/data/output_tracks/1303_29.csv'
    df = pd.read_csv(file_path, sep = ',', header = 0)
    currently_tracked = {}
    in_transition_period = {}
    in_out_of_view_accept_period = {}
    last_frame_num = {}
    prev_frame_num = {}
    first_frame_num = {}
    trajectory = {}
    parallel_list = {}
    for cam_num in range(num_of_camera):
        currently_tracked[cam_num] = False
        in_transition_period[cam_num] = False
        in_out_of_view_accept_period[cam_num] = False
        first_frame_num[cam_num] = -999999999999
        last_frame_num[cam_num] = 99999999999999
        prev_frame_num = 99999999999999
        trajectory[cam_num] = []
        parallel_list[cam_num] = {}
        for other_cam_num in range(num_of_camera):
            if other_cam_num != cam_num:
                parallel_list[cam_num][other_cam_num] = 0
            else:
                parallel_list[cam_num][other_cam_num] = -1
    for index in range(len(df)):
        frame_num = df.iloc[index]['#']
        cam_entry_list = get_cam_entry_list(df, index, num_of_camera)

        for cam_num in range(num_of_camera):
            current_entry = cam_entry_list[cam_num]
            if current_entry == (-1, -1):
                if (frame_num - last_frame_num[cam_num]) == 1:
                    in_out_of_view_accept_period[cam_num] = True
                    in_transition_period[cam_num] = True
                if (frame_num - last_frame_num[cam_num]) > out_of_view_accept_period:
                    in_out_of_view_accept_period[cam_num] = False
                    currently_tracked[cam_num] = False
                if (frame_num - last_frame_num[cam_num]) > transition_period:
                    in_transition_period[cam_num] = False
                    in_out_of_view_accept_period[cam_num] = False

                    first_frame_num[cam_num] = -99999999999
                    last_frame_num[cam_num] = 99999999999
                    prev_frame_num = 999999999999
                    trajectory[cam_num] = []
                
            else:
                if not currently_tracked[cam_num]:
                    first_frame_num[cam_num] = frame_num
                    currently_tracked[cam_num] = True
                    in_out_of_view_accept_period[cam_num] = False
                    in_transition_period[cam_num] = False
                    

                trajectory[cam_num].append((current_entry[1], current_entry[2]))
                last_frame_num[cam_num] = frame_num

            # if (cam_num == 5) and (frame_num in [179832]):
            #     print(frame_num, last_frame_num[cam_num], current_entry, currently_tracked[cam_num], in_transition_period[cam_num], in_out_of_view_accept_period[cam_num])
            # if (cam_num == 6) and (frame_num in [179832]):
            #     print(frame_num, last_frame_num[cam_num], current_entry, currently_tracked[cam_num], in_transition_period[cam_num], in_out_of_view_accept_period[cam_num])

            for other_cam_num in range(num_of_camera):
                if cam_num != other_cam_num:
                    if (currently_tracked[cam_num] == 0) and (currently_tracked[other_cam_num] == 0):
                        parallel_list[cam_num][other_cam_num] = 0
                    elif ((currently_tracked[cam_num] == 1) and (not in_transition_period[cam_num])) and (currently_tracked[other_cam_num] == 1):
                        parallel_list[cam_num][other_cam_num] = 1
                    # if (cam_num == 6) and (other_cam_num == 5) and (frame_num in [179832]):
                    #     print(frame_num, last_frame_num[other_cam_num], current_entry, currently_tracked[other_cam_num], in_transition_period[other_cam_num], in_out_of_view_accept_period[other_cam_num])
                    if in_transition_period[cam_num]:             
                        entry = cam_entry_list[other_cam_num]
                        # if (cam_num == 6) and (other_cam_num == 5) and (frame_num in [179832]):
                        #     print(frame_num, last_frame_num[other_cam_num], current_entry, currently_tracked[other_cam_num], in_transition_period[other_cam_num], in_out_of_view_accept_period[other_cam_num])
                        #     print(parallel_list[cam_num][other_cam_num])
                        # if (other_cam_num == 6) and (frame_num in [179832]):
                        #    print(entry, currently_tracked[other_cam_num], in_transition_period[other_cam_num], in_out_of_view_accept_period[other_cam_num])
                        if (entry != (-1, -1)) and (currently_tracked[other_cam_num]) and (parallel_list[cam_num][other_cam_num] == 0):
                            # print(frame_num, last_frame_num[cam_num], cam_num, other_cam_num, entry, currently_tracked[other_cam_num])
                            traj = np.array([np.array(trajectory[cam_num], dtype = int), other_cam_num, frame_num - last_frame_num[cam_num]], dtype = object)
                            trajectory_list[cam_num].append(traj)

                            currently_tracked[cam_num] = False
                            in_transition_period[cam_num] = False
                            in_out_of_view_accept_period[cam_num] = False
                            first_frame_num[cam_num] = -999999999999
                            last_frame_num[cam_num] = 99999999999999
                            prev_frame_num = 99999999999999
                            trajectory[cam_num] = []
                            break
                    
            
            prev_frame_num = frame_num

for cam_num in trajectory_list.keys():
    if len(trajectory_list[cam_num]) > 0:
        for trajectory in trajectory_list[cam_num]:

            cam_extracted_file_path = os.path.join(extracted_tracks_dir, str(cam_num)) + '.npy'
            with open(cam_extracted_file_path, 'w') as file:
                np.save(cam_extracted_file_path, trajectory)
