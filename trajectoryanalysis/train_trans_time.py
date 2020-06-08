from keras import Sequential, optimizers
from keras.models import load_model
from keras import backend as K
import keras
from keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import preprocessing
import tensorflow as tf
import os
import models
import requests, time, csv, json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print(tf.test.is_gpu_available())

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def slacknoti(contentstr):
    
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

camera = 7
sampling = ["ed", "sw-o", "sw-no", "rw", "irw"]
sampling_id = 4
seq_length = 30
test_slice_part = 80 #! some traces were way too short for this. 60 -> 80

def get_data(camera):
    train = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/npy/train_multi_output/{}.npy".format(camera), allow_pickle=True)
    test = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/npy/test_multi_output/{}.npy".format(camera), allow_pickle=True)

    trainX, trainY = np.hsplit(train,2)
    trainX = np.array([trace[0] for trace in trainX])
    trainY = np.array([trace[0] for trace in trainY])

    testX, testY = np.hsplit(test,2)
    testX = np.array([trace[0] for trace in testX])
    testY = np.array([trace[0] for trace in testY])

    out_dim = np.max(trainY) + 1
    print(out_dim)

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY, out_dim

def gen_data(camera, sampling_id, seq_length, test_slice_part, labeling):
    sampling = ["ed", "sw-o", "sw-no", "rw", "irw"]
    if labeling == 0:
        data = np.load("/home/boyan/out_full_fixed/{}.npy".format(camera), allow_pickle=True)
    elif labeling == 1:
        data = np.load("/home/boyan/AIoTVirt/transitions/traces_train_test.npz", allow_pickle=True)
        
    data = data['cam_{}'.format(camera)]
    train = data[0]
    test = data[1]
    # * Training set
    trainX, trainY, testX, testY = [], [], [], []
    #print(data.shape)
    for series, label, tracelen, transitiontime in train:
        trainX.append(series)
        # y.append(label)
        trainY.append(transitiontime)
        
    for series, label, tracelen, transitiontime in test:
        testX.append(series)
        # y.append(label)
        testY.append(transitiontime)

    trainData = []
    for i, series in enumerate(trainX):
        if len(series) < seq_length:
            continue
        trainData.append(np.array([series, trainY[i]]))

    testData = []
    for i, series in enumerate(testX):
        testData.append(np.array([series, testY[i]]))

    trainX, trainY = preprocessing(trainData, 100, seq_length, sampling[sampling_id])
    # * Testing set
    testX, testY = preprocessing(testData, test_slice_part, seq_length, sampling[sampling_id])
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY, 1


def FCN(seq_length, out_dim):
    model = Sequential()

    model.add(Conv1D(filters=128, kernel_size=8, padding='same', input_shape=(seq_length, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Conv1D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))


    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='linear'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def conv_lstm(seq_length, out_dim):
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=16, padding='same', input_shape=(seq_length, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(seq_length))
    #model.add(Flatten()) #! spencer try
    model.add(Dense(1, activation='linear'))
    model.summary()

    return model



def ResNet(input_shape, nb_classes, n_feature_maps=64):
    input_layer = keras.layers.Input((input_shape,2))

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(1, activation='linear')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
    #             metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

    return model#, reduce_lr

def CNNLSTM(input_shape, out_dim):
    model = Sequential()
    model.add(keras.layers.TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(input_shape)))
    model.add(keras.layers.TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(keras.layers.TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    return model
    

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    

rows = []
names = ["FCN", "CONV_LSTM", "RESNET"]
opts = ["SGD", "RMSProp", "Adagrad", "Adam"]
class_report = {}

outs = ['orig', 'neighbour']
# for m in [conv_lstm, FCN, ResNet]:
for m in [conv_lstm]:
    # data = 
    for camera in range(10):
        print("Processing camera # {}".format(camera))
        sampling = 1
        #trainX, trainY, testX, testY, out_dim = get_data(camera)
        trainX, trainY, testX, testY, out_dim = gen_data(camera, sampling, seq_length, test_slice_part,1)

        optimizer = optimizers.Adam(learning_rate=0.001)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        cam_best_result = 0
        # for i in range(5):
        print("`[MEWTWO]` Training for camera `{}` iteration `{}` at time `{}`".format(camera, 0, time.strftime("%H:%M:%S", time.localtime())))
        # slacknoti("`[MEWTWO]` Training for camera `{}` iteration `{}` at time `{}`".format(camera, i, time.strftime("%H:%M:%S", time.localtime())))
        model = m(seq_length, out_dim)
        # model = models.ResNet((seq_length,2), out_dim)
        model.compile(optimizer=optimizer, loss=root_mean_squared_error)
        #print(trainX.shape, trainY.shape)
        #print(trainY)
        model.fit(trainX, trainY, batch_size=int(len(trainY)/ 5), epochs=150, validation_split=0.15, callbacks=[reduce_lr])
        print(testX.shape, testY.shape)
        result = model.evaluate(testX, testY)
        # slacknoti("`[MEWTWO]` Loss `{}` Accuracy `{}` at time `{}`".format(result[0], result[1], time.strftime("%H:%M:%S", time.localtime())))
        # if result[1] > cam_best_result:
        #     cam_best_result = result[1]
        # model.save('trajectoryanalysis/models/cam_{}.h5'.format(camera))
        class_report[camera] = result
        print(result)

    with open("data/sw-o_lstm_dur+trans_{}.json".format(m.__name__), "w") as file:
        file.write(json.dumps(class_report))
