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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

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
    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY, 1



def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    

rows = []
names = ["FCN", "CONV_LSTM", "RESNET"]
opts = ["SGD", "RMSProp", "Adagrad", "Adam"]
class_report = {}

outs = ['orig', 'neighbour']
# for m in [conv_lstm, FCN, ResNet]:
# for m in [CNNLSTM]:
    # data = 
# for camera in range(10):
sampling_id = 4
seq_length = 30
test_slice_part = 80
#trainX, trainY, testX, testY, out_dim = get_data(camera)
trainX, trainY, testX, testY, out_dim = gen_data(0, sampling_id, seq_length, test_slice_part,1)

model = load_model('trajectoryanalysis/models/cam_0.h5', compile=False)
# print(testX.shape, testY.shape)
for entry, truth in zip(testX[60:66], testY[60:66]):
    pred = model.predict(np.array([entry]))
    print(pred, truth)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001) ,loss=root_mean_squared_error)    
result = model.evaluate(testX[60:66], testY[60:66])
print(result)
# result = model.evaluate(testX, testY)
# class_report[camera] = result
# print(result)

# 32, 1, 16, 9, 22, 16 = 33 + 25 + 38 = 96 / 6 = 16