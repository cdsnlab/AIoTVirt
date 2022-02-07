import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from data_preparation import data_preparation, sampling

from models import ResNet, conv_lstm

import numpy as np
from argparse import ArgumentParser

cam_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

class model_trainer():
    def __init__(
        self, 
        cam_list = cam_list, 
        model = ResNet,
        seq_length = 30,
        sampling_method = ['last', 'ed', 'sw-o'],
        gpu_list = ['0', '1'], 
        data_dir_path = '/home/hihi/carlasim/data/extracted_data/trans_1500_oov_30/training_data'):

        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(physical_devices[1], True)
            tf.config.set_visible_devices(physical_devices[1], 'GPU')
            
        except:
            pass


        self.dataset = data_preparation(data_dir_path)
        self.cam_list = cam_list
        self.gpu_list = gpu_list
        self.sampling_method = sampling_method
        self.used_model = model
        self.seq_length = seq_length

    def train(self):
        for sampling_method in self.sampling_method:
            for cam_num in self.cam_list:
                if cam_num != 6:
                    continue
                print('Training model for cam number {}'.format(cam_num))
                target_list = self.dataset.target_list[cam_num]

                train_trajectories, train_next_cam_label, train_trans_time_label = sampling(method = sampling_method, 
                                                  trajectories = self.dataset.trajectories_train[cam_num],
                                                  next_cam_label = self.dataset.next_cam_label_train[cam_num],
                                                  trans_time_label = self.dataset.trans_time_label_train[cam_num])

                #train_next_cam_label = to_categorical(train_next_cam_label, num_classes = len(target_list))

                val_trajectories, val_next_cam_label, val_trans_time_label = sampling(method = sampling_method, 
                                                  trajectories = self.dataset.trajectories_val[cam_num],
                                                  next_cam_label = self.dataset.next_cam_label_val[cam_num],
                                                  trans_time_label = self.dataset.trans_time_label_val[cam_num])

                #val_next_cam_label = to_categorical(val_next_cam_label, num_classes = len(target_list))

                test_trajectories, test_next_cam_label, test_trans_time_label = sampling(method = sampling_method, 
                                                  trajectories = self.dataset.trajectories_test[cam_num],
                                                  next_cam_label = self.dataset.next_cam_label_test[cam_num],
                                                  trans_time_label = self.dataset.trans_time_label_test[cam_num])


                for task in ['next_cam']:
                    if task == 'next_cam':
                        loss = 'sparse_categorical_crossentropy'
                        metrics = ['accuracy']
                        train_label = train_next_cam_label
                        val_label = val_next_cam_label
                        test_label = test_next_cam_label
                        epoch = 50
                    else:
                        loss = rmse
                        metrics = rmse
                        train_label = train_trans_time_label
                        val_label = val_trans_time_label
                        test_label = test_trans_time_label
                        epoch = 500

                    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.0001)
                    model = self.used_model(input_shape = (self.seq_length, 2), target = task, nb_classes = len(target_list))
                    #model = ResNet(task, len(target_list))
                    model.compile(loss = loss, optimizer = keras.optimizers.SGD(learning_rate = 0.001), metrics = metrics)
                    model.fit(x = train_trajectories, 
                              y = train_label, 
                              batch_size = 16, 
                              epochs = epoch, 
                              validation_data = (test_trajectories, test_label), 
                              callbacks = [reduce_lr]
                              )
                    preds = np.array(model.predict(test_trajectories), dtype = float)
                    print(model.evaluate(test_trajectories, test_label))
                    for i, j in zip(preds, test_label):
                        print(i, j)

            break



def main():
    trainer = model_trainer()
    trainer.train()

if __name__ == '__main__':
    main()


def parse_args():
    args_parser = ArgumentParser()

    args_parser.add_argument(
        '-d', '--data_dir',
        default = '/home/tung/carlasim/data/extracted_tracks/individuals-for-train',
        type = str,
        help = 'The folder that stores the data for training')

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
        '-g', '--gpu_list',
        type = list,
        default = ['0', '1'],
        help = 'The list of GPUs that would be used for training'
    )

    args = args_parser.parse_args()
    return args

