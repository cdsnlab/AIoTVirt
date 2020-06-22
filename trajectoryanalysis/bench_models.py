import matplotlib.pyplot as plt
from statistics import median
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

results = {}

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
#_, _, testX, testY, out_dim = gen_data(camera, sampling, seq_length, test_slice_part,1)
testX = np.random.rand(1, 15, 2)
model = load_model('models/{model}/pp_{pp}_vl_{vl}_cam_{cam}.h5'.format(model="ResNet", pp="ed", vl="15", cam=0), 
                custom_objects={'root_mean_squared_error': root_mean_squared_error})
preds = model.predict(testX)
total = 0
for i in range(1000):
    start = time.time() 
    preds = model.predict(testX)
    total += (time.time() - start)
    
print(total / 1000)
#0.003673448324203491