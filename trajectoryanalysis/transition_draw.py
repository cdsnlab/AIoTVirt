import numpy as np
from preprocessing_time import preprocessing
import tensorflow as tf
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def gen_data(camera, sampling_id, seq_length, test_slice_part, labeling):
    sampling = ["ed", "sw-o", "sw-no", "rw", "irw", "last"]
    if labeling == 0:
        data = np.load("/home/boyan/out_full_fixed/{}.npy".format(camera), allow_pickle=True)
    elif labeling == 1:
        data = np.load("/home/boyan/AIoTVirt/transitions/traces_train_test.npz", allow_pickle=True)
        
    data = data['cam_{}'.format(camera)]
    train = data[0]
    test = data[1]
    # * Training set
    trainX, trainY, testX, testY = [], [], [], []
    testLabel, trainLabel = [], []
    #print(data.shape)
    for series, label, tracelen, transitiontime in train:
        trainX.append(series)
        # y.append(label)
        trainLabel.append(label)
        trainY.append(transitiontime)
        
    for series, label, tracelen, transitiontime in test:
        testX.append(series)
        # y.append(label)
        testLabel.append(label)
        testY.append(transitiontime)

    trainData = []
    trLabels = []
    for i, series in enumerate(trainX):
        if len(series) < seq_length:
            continue
        trainData.append(np.array([series, trainY[i]]))
        trLabels.append(trainLabel[i])

    testData = []
    tsLabels = []
    for i, series in enumerate(testX):
        testData.append(np.array([series, testY[i]]))
        tsLabels.append(testLabel[i])

    _, trainY = preprocessing(trainData, 100, seq_length, sampling[sampling_id])
    # * Testing set
    _, testY = preprocessing(testData, test_slice_part, seq_length, sampling[sampling_id])
    
    # print(len(trainY), len(trLabels))
    trainlabels = {str(label): [] for label in set(trLabels)}
    
    print(set(trLabels))
    for label, transition in zip(trLabels, trainY):
        trainlabels[str(label)].append(transition)
        
    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY, trainlabels



# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true))) 
camera = st.selectbox("Camera", range(10))
    
trainX, trainY, testX, testY, out_dim = gen_data(camera, 5, 30, 80, 1)

# print(type(out_dim.keys()))
target = st.selectbox("Transition to Camera", [-1] + list(out_dim.keys()))
for key, values in out_dim.items():
    if target == -1:
    # plt.plot(range(len(values)), values, label=key)
        plt.plot(values, label=key)
        plt.legend()
    elif key == target:
        plt.plot(values, label=key)
        plt.legend()
# x = range(len(trainY))
st.pyplot()
