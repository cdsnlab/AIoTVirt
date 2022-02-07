import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pickle
from argparse import ArgumentParser
import json
import pandas as pd
from tqdm import tqdm

from traj_sampling import sampling

transition_map = {
    0: [1, 2, 3, 4, 9],
    1: [0, 2, 4, 9],
    2: [0, 1, 9],
    3: [0, 4, 5],
    4: [0, 1, 3, 6],
    5: [3, 7, 8],
    6: [4, 5, 7],
    7: [6, 8, 17],
    8: [5, 6, 7, 17],
    9: [1, 2, 10, 15, 16, 17],
    10: [9, 11, 15],
    11: [10, 12, 14],
    12: [11, 13, 14],
    13: [12, 14],
    14: [11, 12, 13],
    15: [9, 10, 16],
    16: [7, 9, 15, 17],
    17: [7, 9, 16]
}

def args_parser():
    parser = ArgumentParser()

    parser.add_argument(
        '--config_path',
        type = str,
        default = '/home/tung/carlasim/trajectory_analysis_archived/sim_configs/resnet_300_30_bb_15_60.json',
        help = 'Path to the json file that holds the configs for this run.'
    )
    args = parser.parse_args()
    print(args.config_path)

    with open(args.config_path, 'r') as config_file:
        json_args = json.load(config_file)
        config_file.close()
    return json_args

def load_models(args):

    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
            
    except:
        pass

    models = {
        'transtime': {},
        'nextcam': {}
    }
    model_dir_paths = {
        'transtime': os.path.join(args['data_dir_path'], args['transtime_model'], 'models'),
        'nextcam':  os.path.join(args['data_dir_path'], args['nextcam_model'], 'models')
    }

    
    for model_key in models.keys(): #transtime or nextcam
        for cam_num in range(args['num_cams']):
            # if cam_num > 0:
            #     continue
            model_name = '{}_{}_{}_best.h5'.format(
                model_key, 
                'sw-o',
                #args['sampling_method'],
                cam_num
            )
            model_path = os.path.join(model_dir_paths[model_key], model_name)
            if not os.path.exists(model_path):
                continue
            models[model_key][cam_num] = keras.models.load_model(
                model_path,
                compile = False
            )

    return models

def load_sim_data(args):
    sim_data_path = os.path.join(args['data_dir_path'], 'e2e-for-simulation')
    simulation_tracks = {}
    for track_name in os.listdir(sim_data_path):
        track_path = os.path.join(sim_data_path, track_name)
        track = np.load(track_path, allow_pickle = True)
        simulation_tracks[track_name] = track

    return simulation_tracks

def get_transition_time(args):
    min_transition_time = {}
    max_transition_time = {}

    for src_cam_num in range(args['num_cams']):
        for dest_cam_num in transition_map[src_cam_num]:
            min_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] = 9999999999
            max_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] = -9999999999

    track_dir_path = os.path.join(args['data_dir_path'], 'individuals-for-train')
    for file_name in os.listdir(track_dir_path):
        file_path = os.path.join(track_dir_path, file_name)
        cam_recored_tracks = np.load(open(file_path, 'rb'), allow_pickle = True)
        for track in cam_recored_tracks:
            src_cam_num = track[1]
            dest_cam_num = track[2]
            transition_time = track[3]

            try:
                if transition_time < min_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)]:
                    min_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] = transition_time
                if transition_time > max_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)]:
                    max_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] = transition_time
            except:
                pass

    for src_cam_num in range(args['num_cams']):
        for dest_cam_num in transition_map[src_cam_num]:
            if min_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] == 9999999999:
                min_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] = 0
            if max_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] == -9999999999:
                max_transition_time['{}-->{}'.format(src_cam_num, dest_cam_num)] = 0

    return min_transition_time, max_transition_time

def estimate_transition(
    track,
    models,
    tracking_cam,
    curr_frame_idx,
    sampling_method = 'last',
    width = 2560,
    height = 1440
):
    choices = []
    if len(track) < 30:
        return choices

    try:
        sampled_track = sampling(
            method = sampling_method,
            trajectory = track[:, [1, 2]],
            cam_label = None,
            time_label = None,
            sampling_rate = 1,
            seq_length = 30
        )[0]
        for i in range(sampled_track.shape[0]):
            sampled_track[i][0] /= float(width)
            sampled_track[i][1] /= float(height)
        sampled_track = np.expand_dims(np.array(sampled_track, dtype = float), axis = 0)
        
        
        next_cam_preds = models['nextcam'][tracking_cam].predict(sampled_track)
        trans_time_pred = models['transtime'][tracking_cam].predict(sampled_track)
        target_idxs = next_cam_preds[0].argsort()[::-1]
        next_cam_preds = next_cam_preds[0]
        
        for i in target_idxs:
            if next_cam_preds[i] >= 0.25:
                choices.append((transition_map[tracking_cam][i], curr_frame_idx + trans_time_pred))
    except:
        pass
    
    return choices
    


def simulation():

    scenario_name = 'proposed'
    args = args_parser()
    models = load_models(args)
    input()

    
    min_transition_time, max_transition_time = get_transition_time(args)

    simulation_tracks = load_sim_data(args)

    scenario_sim_log_dir_path = os.path.join(args['data_dir_path'], 'scenerio_sim_log')
    os.makedirs(scenario_sim_log_dir_path, exist_ok = True)

    container_boot_time = 0
    num_cams = len(transition_map.keys())

    available_states = ['rec', 'semi-rec', 'trk', 'trans']


    results = []

    total_cam_num_processed_frames = {i: [] for i in range(num_cams)}
    total_cam_num_seen_frames_gt = {i: [] for i in range(num_cams)}
    total_cam_num_seen_frames = {i: [] for i in range(num_cams)}

    reid_time = 20 #miliseconds
    reid_latency_list = []
    

    for track_name in tqdm(simulation_tracks.keys()):
        print(track_name)
        if track_name == '26_244-0.npy':
            continue
    
        # An end-to-end track, whose format should look like this
        # [97958,  1225,   129],     Number of frame in carla, x coordinate, y coordinate
        # [97959,  1225,   129],
        # [97960,  1226,   128],
        # [97961,  1226,   128],
        # [97962,  1226,   127]]) 10 15 98]                Source camera, destination (next) camera, number of transition frames
        # ...
        # [...,  ...,   ...],     
        # [...,  ...,   ...]]) a b c]            
        e2e_track = simulation_tracks[track_name]
        #print(e2e_track)
        state = 'trk'
        tracking_cam = -1
        prev_cam = -1

        trk_timer = 0

        possible_transitions = []
        num_container_frames = {}
        rec_counter ={}

        num_seen_frames = 0
        num_processed_frames = 0
        num_seen_frames_gt = 0

        cam_num_processed_frames = {i: 0 for i in range(num_cams)}
        cam_num_seen_frames_gt = {i: 0 for i in range(num_cams)}
        cam_num_seen_frames = {i: 0 for i in range(num_cams)}
        
        first_seen_frame = -1
        last_seen_frame = -1

        reid_wait = 0


        for entry in e2e_track:

            if reid_wait > 0:
                reid_wait -= 67 #miliseconds
                reid_wait = max(reid_wait, 0)

            frame_idx = entry[0]
            x_coor = entry[1]
            y_coor = entry[2]
            curr_cam = entry[3]

            if curr_cam != -1:
                num_seen_frames_gt += 1
                cam_num_seen_frames_gt[curr_cam] += 1

            if state == 'trk':
                
                if curr_cam != -1:
                    num_seen_frames += 1
                    tracking_cam = curr_cam
                    trk_timer = 0
                    cam_num_seen_frames[tracking_cam] += 1


                    if first_seen_frame == -1:
                        first_seen_frame = frame_idx
                    last_seen_frame = frame_idx
                else:
                    trk_timer += 1
                    if trk_timer > 0:
                        state = 'trans'
                        trk_timer = 0
                        prev_cam = tracking_cam
                        print("[SIM] TRACKING --> TRANSITION at frame number {}".format(frame_idx))

                        possible_transitions = estimate_transition(
                            track = e2e_track[first_seen_frame:(frame_idx)],
                            models = models,
                            tracking_cam = tracking_cam,
                            curr_frame_idx = frame_idx - 1, 
                            sampling_method = args['sampling_method'],
                            width = args['width'],
                            height = args['height']
                        )

                        for neibor, trans_frame in possible_transitions:
                            rec_counter[neibor] = trans_frame

                num_processed_frames += 1
                cam_num_processed_frames[tracking_cam] += 1
                            
            elif state == 'trans':
                for next_cam, trans_frame in possible_transitions:
                    if trans_frame <= frame_idx:
                        num_processed_frames += 1
                        cam_num_processed_frames[next_cam] += 1
                        reid_wait += reid_time
                        reid_latency_list.append(reid_wait)
                        if curr_cam != -1:
                            num_seen_frames += 1
                            

                            state = 'trk'
                            print("[SIM] TRANSITION --> TRACKING at frame number {}".format(frame_idx))
                            tracking_cam = next_cam
                            cam_num_seen_frames[tracking_cam] += 1
                            possible_transitions = []
                            rec_counter = {}
                            first_seen_frame = frame_idx
                            last_seen_frame = frame_idx
                            break
                        else:
                            rec_counter[next_cam] -= 1

                        if rec_counter[next_cam] < 0:
                            possible_transitions.remove((next_cam, trans_frame))
                if (len(possible_transitions) == 0) and state == 'trans': 
                    # for neibor in transition_map[prev_cam]:
                    #     possible_transitions.append(
                    #         neibor,
                    #         int(min_transition_time['{}-->{}'.format(
                    #             prev_cam, 
                    #             neibor
                    #         )]) + frame_idx - 1
                    #     )
                    #     rec_counter[neibor] = 30
                    state = 'rec'
                    print("[SIM] TRANSITION --> RECOVERY at frame number {}".format(frame_idx))
            elif state == 'rec':
                for neibor in transition_map[tracking_cam]:
                    num_processed_frames += 1
                    cam_num_processed_frames[neibor] += 1
                    if curr_cam == neibor:
                        num_seen_frames += 1
                        cam_num_seen_frames[neibor] += 1
                        state = 'trk'
                        print("[SIM] RECOVERY --> TRACKING at frame number {}".format(frame_idx))
                        first_seen_frame = frame_idx
                        last_seen_frame = frame_idx
        results.append((track_name, frame_idx, num_seen_frames_gt, num_processed_frames, num_seen_frames))

        for cam_num in range(num_cams):
            total_cam_num_processed_frames[cam_num].append(cam_num_processed_frames[cam_num])
            total_cam_num_seen_frames_gt[cam_num].append(cam_num_seen_frames_gt[cam_num])
            total_cam_num_seen_frames[cam_num].append(cam_num_seen_frames[cam_num])

    # result_table = pd.DataFrame(
    #     data = results, 
    #     columns = [
    #         'file_name', 
    #         'total_track_frames', 
    #         'num_seen_frames_gt', 
    #         'num_processed_frames', 
    #         'num_seen_frames'
    #     ]
    # )
    # result_table.to_csv(
    #     os.path.join(scenario_sim_log_dir_path, 'proposed.csv'),
    #     index_label = 'index'
    # )

    # total_cam_num_processed_frames = pd.DataFrame.from_dict(
    #     total_cam_num_processed_frames,
    #     orient = 'columns',
    #     #columns = [str(i) for i in range(num_cams)]
    # )
    # total_cam_num_processed_frames.to_csv(
    #     os.path.join(scenario_sim_log_dir_path, '{}_{}.csv'.format(
    #         scenario_name,
    #         'total_cam_num_seen_frames'
    #     )),
    #     index_label = 'index'
    # )
    reid_latency_list_sec = [int(reid_latency / 1000) for reid_latency in reid_latency_list]
    reid_latency_list_sec = pd.DataFrame(reid_latency_list_sec, columns = ['latency'])
    reid_latency_list_sec.to_csv(
        os.path.join(scenario_sim_log_dir_path, '{}_{}.csv'.format(
            scenario_name,
            'reid_latency'
        )),
        index_label = 'index'
    )
        
        


        # for cam_track in simulation_tracks[track_name]:
        #     if len(cam_track[0]) > 30:
        #         sampled_track = sampling(
        #             method = args['sampling_method'],
        #             trajectory = cam_track[0],
        #             cam_label = cam_track[2],
        #             time_label = cam_track[3],
        #             sampling_rate = 1,
        #             seq_length = 30
        #         )
        #         cam_label = cam_track[2]
        #         time_label = cam_track[3]
        #         sampled_track = np.array([sampled_track[0][:, [1, 2]]], dtype = float)
        #         for i in range(sampled_track.shape[1]):
        #             sampled_track[0][i][0] /= float(args['width'])
        #             sampled_track[0][i][1] /= float(args['height'])
        #         print(sampled_track)
        #         print(round(models['transtime'][10].predict(sampled_track).item()), time_label)
        #         break
        # break



if __name__ == '__main__':
    simulation()
    # args = args_parser()
    # simulation_tracks = load_sim_data(args)
    # models = load_models(args)
    # for track_name in simulation_tracks.keys():
    #     for cam_track in simulation_tracks[track_name]:
    #         if len(cam_track[0]) > 30:
    #             sampled_track = sampling(
    #                 method = args['sampling_method'],
    #                 trajectory = cam_track[0],
    #                 cam_label = cam_track[2],
    #                 time_label = cam_track[3],
    #                 sampling_rate = 1,
    #                 seq_length = 30
    #             )
    #             cam_label = cam_track[2]
    #             time_label = cam_track[3]
    #             sampled_track = np.array([sampled_track[0][:, [1, 2]]], dtype = float)
    #             for i in range(sampled_track.shape[1]):
    #                 sampled_track[0][i][0] /= float(args['width'])
    #                 sampled_track[0][i][1] /= float(args['height'])
    #             print(sampled_track)
    #             print(cam_track[1])
    #             #print(round(models['transtime'][cam_track[1]].predict(sampled_track).item()), time_label)
    #             print(models['nextcam'][cam_track[1]].predict(sampled_track), cam_label)
    #             break
    #     input()
    #simulation()
    
    




