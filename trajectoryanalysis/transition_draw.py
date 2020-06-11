import numpy as np
from preprocessing_time import preprocessing
import tensorflow as tf
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def gen_data(camera, sampling_id, seq_length, test_slice_part, labeling, mode):
    sampling = ["ed", "sw-o", "sw-no", "rw", "irw", "last"]
    if labeling == 0:
        data = np.load("/home/boyan/out_full_fixed/{}.npy".format(camera), allow_pickle=True)
    elif labeling == 1:
        data = np.load("/home/boyan/AIoTVirt/transitions/traces_train_test.npz", allow_pickle=True)
        
    predictions = np.load("trajectoryanalysis/models/mae_cam_last_predictions.npz", allow_pickle=True)
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
        if len(series) < seq_length:
            continue
        testData.append(np.array([series, testY[i]]))
        tsLabels.append(testLabel[i])

    trainX, trainY = preprocessing(trainData, 100, seq_length, sampling[sampling_id])
    # * Testing set
    testX, testY = preprocessing(testData, test_slice_part, seq_length, sampling[sampling_id])
    
    # print(len(trainY), len(trLabels))
    # trainlabels = {str(label): {"trans":[], "trace": []} for label in set(trLabels)}
    
    
    
    # print(set(trLabels))
    pred = predictions['cam_{}'.format(camera)]
    # st.dataframe(predictions['cam_1'])
    # print("---")
    print(len(pred))
    print(len(testY))
    if mode == "Test":
        trainlabels = {str(label): {"trans":[], "trace": []} for label in set(tsLabels)}
        print("TestLabels ", trainlabels)
        cnt = 0
        trainY = testY
        print()
        for label, transition in zip(tsLabels, trainY):
            trainlabels[str(label)]["trans"].append(transition)
            try:
                trainlabels[str(label)]["trace"].append(pred[cnt][0])
            except IndexError:
                pass
            cnt +=1
    else:
        trainlabels = {str(label): {"trans":[], "trace": []} for label in set(trLabels)}
        for label, transition in zip(trLabels, trainY):
            trainlabels[str(label)]["trans"].append(transition)
        # print(transition)
        # print(trace)
        
    # model = load_model('trajectoryanalysis/models/mae_cam_last_{}.h5'.format(camera), compile=False)
    # model = m(camera)
    # predictions = model.predict(testX)
    # print(predictions)
    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY, trainlabels



# def root_mean_squared_error(y_true, y_pred):
#         return K.sqrt(K.mean(K.square(y_pred - y_true))) 
camera = st.selectbox("Camera", range(10))
mode = st.radio("Train or Test", ("Train", "Test"))
trainX, trainY, testX, testY, out_dim = gen_data(camera, 5, 30, 80, 1, mode)

# print(type(out_dim.keys()))
target = st.selectbox("Transition to Camera", [-1] + list(out_dim.keys()))


fig = go.Figure()
# for key, (values, _) in out_dim.items():
for key, values in out_dim.items():
    if target == -1:
    # plt.plot(range(len(values)), values, label=key)
        print("###")
        print(len(values["trans"]))
        print(len(values["trace"]))
        fig.add_trace(go.Scatter(x=list(range(len(values["trans"]))), y=values["trans"], mode='lines', name="Grd " + key))
        fig.add_trace(go.Scatter(x=list(range(len(values["trace"]))), y=values["trace"], mode='lines', name="Pred " + key))
    elif key == target:
        fig.add_trace(go.Scatter(x=list(range(len(values["trans"]))), y=values["trans"], mode='lines', name="Grd " + key))
        fig.add_trace(go.Scatter(x=list(range(len(values["trace"]))), y=values["trace"], mode='lines', name="Pred " + key))
        # fig.add_trace(go.Line(values), label=key)
        # plt.plot(values, label=key)
        # plt.legend()
# x = range(len(trainY))
st.plotly_chart(fig, use_container_width=True)
