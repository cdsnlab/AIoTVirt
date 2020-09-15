'''
this program creates non deep learning model based 
'''

from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocessing
#from preprocessing_time import preprocessing
import pandas as pd
from joblib import dump, load
# import tensorflow as tf
import argparse
import os
import models
#from thundersvm import SVC, SVR, NuSVR
import requests, time, csv, json

def slacknoti(contentstr):
    
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})


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

def gen_data(camera, preprocessingmethod, vl, test_slice_part, labeling):
    if labeling == 0:
        data = np.load("/home/boyan/out_full_fixed/{}.npy".format(camera), allow_pickle=True)
    elif labeling == 1:
        data = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/npy/no_dup_label_camera/traces_train_test.npz", allow_pickle=True)
    data = data['cam_{}'.format(camera)]
    train = data[0]
    test = data[1]
    # * Training set
    trainX, trainY, testX, testY = [], [], [], []
    #print(data.shape)
    #! there should be a bucket map tho.
    for series, label, tracelen, transitiontime in train:
        trainX.append(series)
        trainY.append(label)
        
    for series, label, tracelen, transitiontime in test:
        testX.append(series)
        testY.append(label)
        
    trainData = []
    for i, series in enumerate(trainX):
        if len(series) < vl:
            continue
        trainData.append(np.array([series, trainY[i]])) 

    testData = []
    for i, series in enumerate(testX):
        testData.append(np.array([series, testY[i]]))
    trainX, trainY = preprocessing(trainData, 100, vl, preprocessingmethod)
    # * Testing set
    testX, testY = preprocessing(testData, test_slice_part, vl, preprocessingmethod)

    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY, 1


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
pp_looper=['last', 'ed', 'irw', 'sw-o']
vl_looper=[15, 30]
meth_looper=['dt', 'rf', 'svm']

#! for testing each cases
# vl_looper=[15]
# pp_looper=['irw']
# meth_looper=['dt']

test_slice_part = 80 #! some traces were way too short for this. 60 -> 80
name = 0
results_for_excel = pd.DataFrame(
    columns = ["timemodel", "preprocessingmethod", "vl", "Camera 0", "Camera 1", "Camera 2", "Camera 3", "Camera 4", "Camera 5", "Camera 6", "Camera 7", "Camera 8", "Camera 9"]
)


for timemodel in meth_looper:
    for vl in vl_looper:
        for pp in pp_looper:
            final={}
            for camera in range(10):
                print("Processing camera # {}".format(camera))
                #trainX, trainY, testX, testY, out_dim = get_data(camera)
                trainX, trainY, testX, testY, out_dim = gen_data(camera, pp, vl, test_slice_part,1) #* 1:sw-o or 4:irw
                #print(trainX.shape, trainY.shape)

                print("`[MEWTWO]` Training for camera `{}` pp {}, vl {}, meth {} at time `{}`".format(camera, pp, vl, timemodel, time.strftime("%H:%M:%S", time.localtime())))
                # slacknoti("`[MEWTWO]` Training for camera `{}` iteration `{}` at time `{}`".format(camera, i, time.strftime("%H:%M:%S", time.localtime())))
                #clf=SVR(gpu_id=0)
                if timemodel =="dt":
                    clf=DecisionTreeClassifier(max_depth=10)
                elif timemodel=="rf":
                    clf=RandomForestClassifier(n_jobs=20)
                elif timemodel=="svm":
                    clf=SVC()
                
                trainsamples, trainnx, trainny = trainX.shape
                testsamples, testnx, testny = testX.shape
                d2_trainX_dataset = trainX.reshape(len(trainX), vl*2)
                
                clf.fit(d2_trainX_dataset, trainY)
                d2_testX_dataset = testX.reshape(len(testX), vl*2)
                
                #predy=clf.predict(d2_testX_dataset)
                
                predict_label = clf.score(d2_testX_dataset, testY)
                print ('camera {} test accuracy {:.2f}'.format(camera, predict_label))
                # print(testY)
                # print(predict_label)
                #print(predy)
                #final[camera]=mean_squared_error(testY, predy, squared=False)
                final[camera]=predict_label
                #print ('++++++++++++++camera {} test RMSE {:.2f}'.format(camera, final[camera]))
                dump(clf, 'joblibs_tr_dir/{}/pp_{}_vl_{}_cam_{}.joblib'.format(timemodel, pp, vl, camera))
                print('[INFO] dumped model')
            results_for_excel = results_for_excel.append(pd.Series(data=[timemodel, pp, vl, final[0], final[1], final[2], final[3], final[4], final[5], final[6], final[7], final[8],final[9]], index=results_for_excel.columns, name=name))
            name+=1
    slacknoti("done timemodel: {}, at {}".format(timemodel, time.strftime("%H:%M:%S", time.localtime())))
with pd.ExcelWriter("dir_est_results/dir_est.xlsx", mode='w') as writer:
    results_for_excel.to_excel(writer, sheet_name="mama")
