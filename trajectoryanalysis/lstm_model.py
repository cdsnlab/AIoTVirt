from keras import Sequential, optimizers
from keras.models import load_model
import keras
from keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, MaxPooling1D
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import preprocessing
import tensorflow as tf
import os
import models
import requests, time, csv, json

# print(tf.test.is_gpu_available())

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def slacknoti(contentstr):
    
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

camera = 7
sampling = ["ed", "sw-o", "sw-no", "rw", "irw"]
sampling_id = 4
seq_length = 30
test_slice_part = 60

def get_data(camera):
    #train = np.load("/home/boyan/new_labeling/train_{}.npy".format(camera), allow_pickle=True)
    #test = np.load("/home/boyan/new_labeling/test_{}.npy".format(camera), allow_pickle=True)

    train = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/samplednpy/train_{}.npy".format(camera), allow_pickle=True)
    test = np.load("/home/spencer1/AIoTVirt/trajectoryanalysis/samplednpy/test_{}.npy".format(camera), allow_pickle=True)


    trainX, trainY = np.hsplit(train,2)
    trainX = np.array([trace[0] for trace in trainX])
    trainY = np.array([trace[0] for trace in trainY])

    testX, testY = np.hsplit(test,2)
    testX = np.array([trace[0] for trace in testX])
    testY = np.array([trace[0] for trace in testY])

    out_dim = np.max(trainY) + 1

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY, out_dim

def gen_data(camera, sampling_id, seq_length, test_slice_part, labeling):
    sampling = ["ed", "sw-o", "sw-no", "rw", "irw"]
    if labeling == 0:
        data = np.load("/home/boyan/out_full_fixed/{}.npy".format(camera), allow_pickle=True)
    elif labeling == 1:
        data = np.load("/home/boyan/model_out_label_neighb/{}.npy".format(camera), allow_pickle=True)
    # print(data[0][0].shape)
    # * Training set
    x, y = [], []
    # print(data.shape)
    for series, label in data:
        x.append(series)
        y.append(label)

    out_dim = max(set(y)) + 1
    y = to_categorical(y)    
    trainX, testX, trainY, testY = train_test_split(np.array(x), np.array(y), test_size=0.20, shuffle=False)

    trainData = []
    for i, series in enumerate(trainX):
        if len(series) < seq_length:
            continue
        trainData.append(np.array([series, trainY[i]]))

    testData = []
    for i, series in enumerate(testX):
        testData.append(np.array([series, testY[i]]))

    trainX, trainY = preprocessing(trainData, 100, seq_length, sampling[sampling_id])
    # trainX, _, trainY, _  = train_test_split(X, Y, test_size=0.25, shuffle=False)
    # * Testing set
    testX, testY = preprocessing(testData, test_slice_part, seq_length, sampling[sampling_id])
    # _, testX, _, testY = train_test_split(X, Y, test_size=0.25, shuffle=False)

    # out_dim = max(set(trainY)) + 1
    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
    return trainX, trainY, testX, testY, out_dim


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
    model.add(Dense(out_dim, activation='softmax'))
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def conv_lstm(seq_length, out_dim):
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=16, padding='same', input_shape=(seq_length, 2)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(seq_length))
    model.add(Dense(out_dim, activation='softmax'))

    return model

optimizer = [
    optimizers.SGD(learning_rate=0.01),
    optimizers.RMSprop(learning_rate=0.01),
    optimizers.Adagrad(learning_rate=0.01),
    optimizers.Adam(learning_rate=0.01)
]
# optimizer = [optimizers.RMSprop(learning_rate=0.01)]
results = []
# for opt in optimizer:
#     print("Training {}".format(opt))
#     # model = FCN(seq_length, out_dim)
#     model = conv_lstm(seq_length, out_dim)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     history = model.fit(trainX, trainY, batch_size=100, epochs=30, validation_split=0.1)

#     results.append(model.evaluate(testX, testY))

# while True:
#     model = conv_lstm(seq_length, out_dim)
#     model.compile(optimizer=optimizer[0], loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit(trainX, trainY, batch_size=100, epochs=30, validation_split=0.1)
#     result = model.evaluate(testX, testY)
#     if result[1] > 0.76:
#         model.save('conv_lstm_rw.h5')
#         break
rows = []
names = ["FCN", "CONV_LSTM", "RESNET"]
opts = ["SGD", "RMSProp", "Adagrad", "Adam"]
class_report = {}

outs = ['orig', 'neighbour']
for labeling in [1]:
    # data = 
    for camera in range(0,10):
        sampling = 4
        trainX, trainY, testX, testY, out_dim = get_data(camera)
        optimizer = optimizers.Adam(learning_rate=0.001)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        cam_best_result = 0
        for i in range(5):
            slacknoti("`[MEWTWO]` Training for camera `{}` iteration `{}` at time `{}`".format(camera, i, time.strftime("%H:%M:%S", time.localtime())))
            model = conv_lstm(seq_length, out_dim)
            # model = models.ResNet((seq_length,2), out_dim)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(trainX, trainY, batch_size=int(len(trainY)/ 5), epochs=150, validation_split=0.15, callbacks=[reduce_lr])
            # print(testX)
            result = model.evaluate(testX, testY)
            slacknoti("`[MEWTWO]` Loss `{}` Accuracy `{}` at time `{}`".format(result[0], result[1], time.strftime("%H:%M:%S", time.localtime())))
            if result[1] > cam_best_result:
                cam_best_result = result[1]
                model.save('model_new_label/cam_{}.h5'.format(camera))
        class_report[camera] = cam_best_result

    with open("results/lstm_new_label.json", "w") as file:
        file.write(json.dumps(class_report))

# for camera in [8,7,4,1,0]:
#     best_result = 0
#     for sampling_id in range(4):
#         trainX, trainY, testX, testY, out_dim = gen_data(camera, sampling_id, seq_length, test_slice_part)
#         for i, model in enumerate([FCN(seq_length, out_dim), conv_lstm(seq_length, out_dim), models.ResNet((seq_length, 2), out_dim)]):
#             for j, opt in enumerate(optimizer):
#                 model.compile(optimizer=optimizer[0], loss='categorical_crossentropy', metrics=['accuracy'])
#                 for n in range(5):
#                     model.fit(trainX, trainY, batch_size=int(len(trainY) / 40), epochs=30, validation_split=0.1)
#                     result = model.evaluate(testX, testY)
#                     slacknoti("`[Mewtwo] [Deep Learning]` Model {} on camera `{}` complete at time {} with params sampling `{}` and optimizer `{}` \n Loss {} Accuracy {}".format(names[i], camera, time.strftime("%H:%M:%S", time.localtime()), sampling[sampling_id], opts[j], result[0], result[1]))
#                     rows.append([camera, names[i], sampling[sampling_id], opts[j], result[0], result[1]])
#                     if result[1] > best_result:
#                         best_result = result[1]
#                         model.save("models/cam_{}_sampling_{}_model_{}_opt_{}.h5".format(camera, sampling[sampling_id], names[i], opts[j]))

# with open('results.csv', mode='w') as file:
#     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     headers = ['Camera', 'Model', 'Sampling', 'Optimizer', 'Loss', 'Accuracy']
#     writer.writerow(headers)
#     for row in rows:
#         writer.writerow(row)

# m = load_model('conv_lstm_rw.h5')
# print(m.evaluate(testX, testY))
# # model, callback = models.ResNet((seq_length, 2), out_dim)
# # model.fit(trainX, trainY, batch_size=40, epochs=30, validation_split=0.1, callbacks=[callback])
# # print(model.evaluate(testX, testY))
# # print(results, "FCN")
# print(results, "CONV_LSTM")