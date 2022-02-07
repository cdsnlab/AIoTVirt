import os
import logging
from types import new_class

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from keras.utils import tf_utils, io_utils
from tensorflow.python.platform import tf_logging as logging


from callbacks import CustomModelCheckPoint, Logging, EarlyStopping

from data_preparation import data_preparation, sampling

from models import ResNet, conv_lstm, transformer

import numpy as np
from argparse import ArgumentParser
from math import ceil
import json

#cam_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
cam_list = [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

model_dict = {
    'convlstm': conv_lstm,
    'resnet': ResNet,
    'transformer': transformer
}

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

class model_trainer():
    def __init__(
        self, 
        cam_list = cam_list, 
        model_name = 'transformer',
        seq_length = 30,
        sampling_method = ['last', 'ed', 'sw-o'],
        gpu_num = 0, 
        data_dir_path = '/home/hihi/carlasim/data_archived/extracted_tracks_300_30_bb_15_60',
        tconfig_path = '/home/hihi/carlasim/trajectory_analysis_archived/sim_configs/transformer.json'
    ):

        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[gpu_num], True)
            tf.config.set_visible_devices(physical_devices[gpu_num], 'GPU')
            
        except:
            pass


        self.cam_list = cam_list
        self.gpu_num = gpu_num
        self.sampling_method = sampling_method
        self.model_name = model_name
        self.model = model_dict[model_name]
        self.seq_length = seq_length

        self.tconfigs = json.load(open(tconfig_path, 'r')) # configs for the transformer
        if model_name == 'transformer':
            self.model_name += '_{}_{}_{}_{}_{}'.format(
                self.tconfigs['head_size'],
                self.tconfigs['num_heads'],
                self.tconfigs['ff_dim'],
                self.tconfigs['num_transformer_blocks'],
                self.tconfigs['mlp_units']
            )

        self.train_data_path = os.path.join(data_dir_path, 'individuals-for-train')
        self.save_model_path = os.path.join(data_dir_path, self.model_name, 'models')
        os.makedirs(self.save_model_path, exist_ok = True)
        self.test_log_path = os.path.join(data_dir_path, self.model_name, 'test_logs')
        os.makedirs(self.test_log_path, exist_ok = True)
        self.train_log_path = os.path.join(data_dir_path, self.model_name, 'train_logs')
        os.makedirs(self.train_log_path, exist_ok = True)
        self.target_list = {
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

        self.dataset = data_preparation(self.train_data_path)

    def run(self, train = True, test = True):
        for sampling_method in self.sampling_method:
            if sampling_method != 'sw-o':
                continue
            for cam_num in self.cam_list:
                if cam_num < 7:
                    continue
                target_list = self.target_list[cam_num]

                train_trajectories, train_next_cam_label, train_trans_time_label = sampling(method = sampling_method, 
                                                  trajectories = self.dataset.trajectories_train[cam_num],
                                                  next_cam_label = self.dataset.next_cam_label_train[cam_num],
                                                  trans_time_label = self.dataset.trans_time_label_train[cam_num])
                val_trajectories, val_next_cam_label, val_trans_time_label = sampling(method = sampling_method, 
                                                  trajectories = self.dataset.trajectories_val[cam_num],
                                                  next_cam_label = self.dataset.next_cam_label_val[cam_num],
                                                  trans_time_label = self.dataset.trans_time_label_val[cam_num])

                test_trajectories, test_next_cam_label, test_trans_time_label = sampling(method = sampling_method, 
                                                  trajectories = self.dataset.trajectories_test[cam_num],
                                                  next_cam_label = self.dataset.next_cam_label_test[cam_num],
                                                  trans_time_label = self.dataset.trans_time_label_test[cam_num])
                # print(test_trajectories, test_next_cam_label, target_list)
                # print(train_next_cam_label, test_next_cam_label)

                for task in ['nextcam', 'transtime']:
                    if task == 'nextcam':
                        loss = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                        train_label = train_next_cam_label
                        test_label = test_next_cam_label
                        metric = 'val_accuracy'
                        epoch = 100
                        #continue
                    else:
                        loss = rmse
                        metrics = rmse
                        train_label = train_trans_time_label
                        test_label = test_trans_time_label
                        metric = 'val_rmse'
                        epoch = 500
                        #continue
                    

                    run_name = '{}_{}_{}'.format(task, sampling_method, cam_num)
                    if train:
                        print('Training [{}] for cam [{}] with sampling method [{}]'.format(task, cam_num, sampling_method))
                        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.0001)
                        model_check_point = CustomModelCheckPoint(
                            file_path = os.path.join(self.save_model_path, run_name + '_{epoch}'),
                            metric = metric
                        )
                        train_logging = Logging(
                            log_path = os.path.join(self.train_log_path, run_name)
                        )
                        early_stopping = EarlyStopping(
                            monitor = 'val_accuracy', 
                            min_delta = 0.03, 
                            patience = 2, 
                            mode = 'max',
                            minimum = 0.9
                        )
                        if 'transformer' not in self.model_name:
                            model = self.model(input_shape = (self.seq_length, 2), target = task, nb_classes = len(target_list))
                            model.compile(loss = loss, optimizer = keras.optimizers.Adam(learning_rate = 0.0001), metrics = metrics)
                        else:
                            
                            tconfigs = self.tconfigs #configs for the transformer
                            model = self.model(
                                task = task,
                                n_classes = len(target_list),
                                input_shape = (self.seq_length, 2),
                                head_size = tconfigs['head_size'],
                                num_heads = tconfigs['num_heads'],
                                ff_dim=tconfigs['ff_dim'],
                                num_transformer_blocks=tconfigs['num_transformer_blocks'],
                                mlp_units=[tconfigs['mlp_units']],
                                mlp_dropout=tconfigs['mlp_dropout'],
                                dropout=tconfigs['dropout'],

                            )
                            model.compile(loss = loss, optimizer = keras.optimizers.SGD(learning_rate = 0.001), metrics = metrics)
                        model.fit(
                            x = train_trajectories, 
                            y = train_label, 
                            batch_size = 16, 
                            epochs = epoch, 
                            validation_data = (test_trajectories, test_label), 
                            callbacks = [reduce_lr, train_logging, model_check_point, early_stopping],
                            verbose = True,
                        )
                        #model.evaluate(test_trajectories, test_label)
                    if test:
                        print('Testing [{}] for cam [{}] with sampling method [{}]'.format(task, cam_num, sampling_method))
                        model = keras.models.load_model(
                            os.path.join(self.save_model_path, run_name + '_best.h5'),
                            compile = False
                        )
                        model.compile(loss = loss, optimizer = keras.optimizers.SGD(learning_rate = 0.001), metrics = metrics)

                        if task == 'nextcam':
                            log_path = os.path.join(
                                self.test_log_path,
                                run_name + '_accuracy.txt'
                            )
                            #print(model.evaluate(test_trajectories, test_label))
                            accuracy = float(model.evaluate(test_trajectories, test_label)[1])
                            with open(log_path, 'w') as log_file:
                                log_file.write('{}'.format(accuracy))
                                log_file.close()
                        else:
                            log_path = os.path.join(
                                self.test_log_path,
                                run_name + '_frame_loss.csv'
                            )
                            preds = np.array(model.predict(train_trajectories), dtype = float)
                            with open(log_path, 'w') as log_file:
                                log_file.write('pred,label,loss\n')
                                for i, j in zip(preds, train_label):
                                    log_file.write('{},{},{}\n'.format(
                                        float(i), 
                                        j, 
                                        int(ceil(i - j)) # negative if early prediction
                                                        # positive if late prediction
                                    ))
                    print('DONE')




def parse_args():
    args_parser = ArgumentParser()

    args_parser.add_argument(
        '-d', '--data_dir',
        default = '/home/tung/carlasim/data/extracted_tracks/individuals-for-train',
        type = str,
        help = 'The folder that stores the data for training')

    args_parser.add_argument(
        '--tconfig_path',
        default = '/home/hihi/carlasim/trajectory_analysis_archived/sim_configs/transformer.json',
        type = str
    )

    args_parser.add_argument(
        '--lr', '--learning_rate',
        type = float,
        help = 'Learning for the models'
    )

    args_parser.add_argument(
        '-c', '--cam_list',
        type = list,
        help = 'The list of cameras that would be trained'
    )

    args_parser.add_argument(
        '-g', '--gpu_num',
        type = int,
        default = 0,
        help = 'The list of GPUs that would be used for training'
    )

    args = args_parser.parse_args()
    return args

def main():
    args = parse_args()
    trainer = model_trainer(
        gpu_num = args.gpu_num,
        tconfig_path = args.tconfig_path
    )
    trainer.run(test = True)

if __name__ == '__main__':
    main()



