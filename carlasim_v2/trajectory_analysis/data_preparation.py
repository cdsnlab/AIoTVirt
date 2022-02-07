import os
import numpy as np
import math

class data_preparation():
    def __init__(self, data_dir_path = '/home/tung/carlasim/data/extracted_tracks/individuals-for-train', num_of_cam = 18, train_size = 0.6, val_size = 0, seed = 700, frame_width = 2560, frame_height = 1440, frame_rate = 15):
        
        self.num_of_cam = num_of_cam
        self.data_dir_path = data_dir_path
        self.train_size = train_size
        self.val_size = val_size
        self.seed = seed
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_rate = frame_rate
        self.cam_list = []

        self.data_train = {}
        self.data_test = {}
        self.data_val = {}
        self.trajectories_train = {}
        self.next_cam_label_train = {}
        self.trans_time_label_train = {}
        self.trajectories_test = {}
        self.next_cam_label_test = {}
        self.trans_time_label_test = {}
        self.trajectories_val = {}
        self.next_cam_label_val = {}
        self.trans_time_label_val = {}
        self.target_list = {}
        self.e_list = {
            0: [15, 17],
            1: [],
            2: [],
            3: [],
            4: [2],
            5: [],
            6: [],
            7: [16],
            8: [],
            9: [0, 16],
            10: [],
            11: [],
            12: [],
            13: [],
            14: [],
            15: [],
            16: [],
            17: [0, 15]
        }

        self.load_data()

        self.train_test_split()

        
    def load_data(self):
        self.dataset = {}
        for file in os.listdir(self.data_dir_path):
            if '.npy' in file:
                cam_num = int(file.split('.')[0])
                file_path = os.path.join(self.data_dir_path, file)
                self.dataset[cam_num] = np.load(file_path, allow_pickle = True)
                self.cam_list.append(cam_num)
                self.target_list[cam_num] = []
        self.cam_list.sort()
    def train_test_split(self):

        np.random.seed(self.seed)

        def split(data, cam_num):
            coors = []
            next_cam_label = []
            trans_time_label = []

            for series, _, next_cam, trans_time in data:
                if next_cam not in self.e_list[cam_num]:
                    coor = []
                    first_frame = 0
                    for i, (frame, x, y) in enumerate(series):
                        if i == 0:
                            first_frame = frame
                        coor.append((x / self.frame_width, y / self.frame_height))

                    coors.append(coor)
                    next_cam_label.append(next_cam)
                    trans_time_label.append(trans_time)

                    if next_cam not in self.target_list[cam_num]:
                        self.target_list[cam_num].append(next_cam)
            self.target_list[cam_num].sort()


            return coors, next_cam_label, trans_time_label     

        for cam_num in self.cam_list:
            np.random.seed(self.seed)
            np.random.shuffle(self.dataset[cam_num])
            num_of_train_samples = int(self.train_size * len(self.dataset[cam_num]))
            num_of_val_samples = int(self.val_size * len(self.dataset[cam_num]))

            self.data_train[cam_num] = self.dataset[cam_num][:num_of_train_samples]
            self.data_val[cam_num] = self.dataset[cam_num][num_of_train_samples:(num_of_train_samples + num_of_val_samples)]
            self.data_test[cam_num] = self.dataset[cam_num][:]#[num_of_train_samples + num_of_val_samples:]

            #print(cam_num)
            #print(len(self.data_train[cam_num]), len(self.data_val[cam_num]), len(self.data_test[cam_num]))

            self.trajectories_train[cam_num], self.next_cam_label_train[cam_num], self.trans_time_label_train[cam_num] = split(self.data_train[cam_num], cam_num)
            self.trajectories_val[cam_num], self.next_cam_label_val[cam_num], self.trans_time_label_val[cam_num] = split(self.data_val[cam_num], cam_num)
            self.trajectories_test[cam_num], self.next_cam_label_test[cam_num], self.trans_time_label_test[cam_num] = split(self.data_test[cam_num], cam_num)

            self.next_cam_label_train[cam_num] = [self.target_list[cam_num].index(x) for x in self.next_cam_label_train[cam_num]]
            self.next_cam_label_val[cam_num] = [self.target_list[cam_num].index(x) for x in self.next_cam_label_val[cam_num]]
            self.next_cam_label_test[cam_num] = [self.target_list[cam_num].index(x) for x in self.next_cam_label_test[cam_num]]
            #print(list(set(self.next_cam_label_train[cam_num])))

            #print(len(self.trajectories_train[cam_num]), len(self.trajectories_val[cam_num]), len(self.trajectories_test[cam_num]))

            #print(cam_num, self.target_list[cam_num])
def sampling(
    method, 
    trajectories, 
    next_cam_label, 
    trans_time_label, 
    sampling_rate = 1, 
    seq_length = 30
):

    sampled_trajectories = []
    sampled_cam_label = []
    sampled_time_label = []
    for trajectory, cam_label, time_label in zip(trajectories, next_cam_label, trans_time_label):
        if method == 'sw-o': #Sliding window complete overlap
            if len(trajectory) > 60:
                trajectory = trajectory[-60:]
            portion_len = int(len(trajectory) * sampling_rate)

            for i, coordinates in enumerate(trajectory):
                if (i + seq_length) > portion_len:
                    break
                sampled_trajectories.append(trajectory[i : (i + seq_length)])
                sampled_cam_label.append(cam_label)
                sampled_time_label.append(time_label)
    
        # elif method == 'sw-no': #Sliding window without overlapping
        #     for i, (x, y) in enumerate(zip(XX, YY)):
        #         if (i % seq_length) == 0:
        #             if (i + seq_length) > portion_len:
        #                 break
        #             sampled_X.append(XX[i : (i + seq_length)])
        #             sampled_Y.append(YY[i : (i + seq_length)])
        #             sampled_cam_label.append(cam_label)
        #             sampled_time_label.append(time_label)
        #         else:
        #             pass

        elif (method == 'ed') or ((method == 'irw') and (sampling_rate != 1)): # Evenly distributed
            if len(trajectory) > 60:
                trajectory = trajectory[-60:]
            portion_len = int(len(trajectory) * sampling_rate)
            tmp_traj = []
            count = 0
            btw = math.floor(portion_len / float(seq_length))
            if portion_len < seq_length:
                continue
            for coors in trajectory[::btw]:
                if count == seq_length:
                    break
                else:
                    tmp_traj.append(coors)
                count += 1
            sampled_trajectories.append(tmp_traj)
            sampled_cam_label.append(cam_label)
            sampled_time_label.append(time_label)
        # elif (method == 'irw'): #inverse reducing windows
        #     portions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #     rwset = [0.9, 0.8, 0.7]
        #     for prt in portions:
        #         start_loc = int(len(XX) / prt)
        #         for rw in rwset:
        #             end_loc = int(len(XX) / prt)
        #             btw = math.floor((end_loc - start_loc) / float(seq_length))
        #             tmp_x, tmp_y = [], []
        #             count = 0
        #             if (end_loc - start_loc) < seq_length:
        #                 continue
        #             for x, y in zip(XX[start_loc:end_loc:btw], YY[start_loc:end_loc:btw]):
        #                 if count == seq_length:
        #                     break
        #                 else:
        #                     tmp_x.append(x)
        #                     tmp_y.append(y)
        #                 count += 1
        #             sampled_X.append(tmp_x)
        #             sampled_Y.append(tmp_y)
        #             sampled_cam_label.append(cam_label)
        #             sampled_time_label.append(time_label)
                        
        elif (method == 'last') or ((method == 'sw-o') and (sampling_rate != 1)):
            portion_len = int(len(trajectory) * sampling_rate)
            if len(trajectory) < seq_length:
                continue
            sampled_trajectories.append(trajectory[-seq_length:])
            sampled_cam_label.append(cam_label)
            sampled_time_label.append(time_label)

    return np.array(sampled_trajectories), np.array(sampled_cam_label), np.array(sampled_time_label, dtype = np.float32)
        
# a = data_preparation()
# for cam_num in a.cam_list:
#     temp_traj, temp_cam_label, temp_time_label = sampling('sw-o', a.trajectories_train[cam_num], a.next_cam_label_train[cam_num], a.trans_time_label_train[cam_num])  
#     #print(a.trajectories_test[cam_num][4])
#     print(temp_traj.shape, temp_cam_label.shape, temp_time_label.shape)
    #print(temp_traj[4])
    #break
#     for i, (traj, cam_label, time_label) in enumerate(zip(temp_traj, temp_cam_label, temp_time_label)):
#         if (len(traj) != 30):
#             print(cam_num, i)

# print(len(a.trajectories_test[0]))
# temp_traj, temp_cam_label, temp_time_label = sampling('ed', a.trajectories_test[0], a.next_cam_label_test[0], a.trans_time_label_test[0])
# print(len(temp_traj))
# for i in temp_traj:
#     if len(i) != 30:
#         print(len(i))

#b = np.array(a.trajectories_train[6])




        

