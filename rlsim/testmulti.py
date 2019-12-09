#!/usr/bin/env python

import logging
import logging.handlers
import gym
import sys
import time
import random
#import camera
import camera4
import cv2
import _pickle as pickle
from env_4cam import chamEnv
from draw import drawgraph
import csv
import requests
import json
import argparse
import configparser
import os
import threading

vidpath = "/home/spencer/samplevideo/testedvids/"
c = []
cidx=[]
th=[]
for i in range(0,4):
# load videos and process frame by frame.
    c.append(camera4.cam(str(i)))
    #th.append(threading.Thread(target=c[i].procframe, args=(c[i].id, 0)))

start_time = time.time()

for k in range(0, int(c[0].cap.get(cv2.CAP_PROP_FRAME_COUNT)-2)):
    for i in range(0, 4):
        #cidx.append(c[i].procframe(c[i].id, k))
        #my_thread = threading.Thread(target=c[i].procframe, args=(c[i].id, k))
        th.append(threading.Thread(target=c[i].procframe, args=(c[i].id, 0)))
        #print("up",i)
        th[i].start()
        #my_thread.join()
    for i in th:
        #cidx.append(c[i].procframe(c[i].id, k))
        #my_thread = threading.Thread(target=c[i].procframe, args=(c[i].id, k))
        #print(i)
        i.join()
    th=[]
end_time = time.time()
print(end_time-start_time)


#def run_thr(target):
#    res = target.procframe
#    cidx.append(res)
        
