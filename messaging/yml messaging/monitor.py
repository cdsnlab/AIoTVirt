# need to make this into a request handler version.

import time, sys, os, json, torch
import random
import zmq
import cv2
import argparse
import pickle as pkl
import numpy as np
from reid import reID
from yolo_detector import Detector
from efficientdet.detection import Detection
import math

parser = argparse.ArgumentParser(description="Run local scenario")
parser.add_argument('-t', '--template', type=str, default="/home/spencer1/samplevideo/for_demo/duke/duke_reid_0.JPG", help='Path to template image to use for reid matching')
#parser.add_argument('-t', '--template', type=str, default="/home/spencer1/samplevideo/for_demo/camnet/camnet_reid_0.JPG", help='Path to template image to use for reid matching')
#parser.add_argument('-t', '--template', type=str, default="/home/spencer1/samplevideo/for_demo/virat/virat_reid_0.JPG", help='Path to template image to use for reid matching')
#parser.add_argument('-t', '--template', type=str, default="/home/spencer1/samplevideo/for_demo/carla/carla_reid_0.JPG", help='Path to template image to use for reid matching')
parser.add_argument('--threshold', type=float, default=0.30, help='Threshold value for reID similarity. Lower is closer.Default is 0.35')
parser.add_argument('--destination',required=False, default= 'None')
parser.add_argument('--start_time',required=False, default='0')
parser.add_argument('--sector',required=False, default='0')


args = parser.parse_args()

time.sleep(int(args.start_time))


############ model setup here
# * Init Detector
detector = Detection(0)
detector.load_model("/home/boyan/edge-scheme-demo/efficientdet/weights/efficientdet-d")

# * yolo Detector
# detector = Detector(0) #! segmentation fault here. ditch or get it from pytorch yolo repository.

context = zmq.Context()

total_num_cam = 1

############ zmq setup here
#TODO setup the sockets so that its identifiable 
# 5556~8 is for client -> server re-id request channel
socket_reid_req_0 = context.socket(zmq.PULL)
socket_reid_req_0.bind("tcp://*:5556")

socket_reid_req_1 = context.socket(zmq.PULL)
socket_reid_req_1.bind("tcp://*:5557")

socket_reid_req_2 = context.socket(zmq.PULL)
socket_reid_req_2.bind("tcp://*:5558")


# 5555 is for server -> client re-id reply channel
socket_reid_rep = context.socket(zmq.PUSH)
socket_reid_rep.bind("tcp://*:5555")
#socket_reid_req_0.setsockopt(zmq.RCVTIMEO, 5000)

print("[INFO] zmq ready")

time_now = time.time()
time_before = time.time()

#TODO we need to iterate through sockets and return all replies.
'''
while True:
    #  Wait for next request from client
    message = socket_reid_req.recv_pyobj()
    time_before = time_now
    time_now = time.time()
    frame= message['frame']
    count= message['count']
    send_time = message['send_time']
    wc = message['wc']
    input_fps = message['input_fps']

    if wc == "EOF":
        print("[INFO] finishing at count {}".format(count))
        socket_reid_rep.send_pyobj(dict(camnum=camnum,frame=frame, ts=time.time(), count=count, send_time=send_time, wc="EOF"))
        continue

    if count==0:
        print("[INFO] received request: count {}".format(count))
    else:
        print("[INFO] received request: count {},input fps {}, process fps : {}".format(count,input_fps,1/(time_now-time_before)))
    # do object detection here. 
    detections, boundaries = detector.detect(frame)
    for i in range(len(detections)):
        print('[INFO] Object speed {} pixel per second'.format(math.sqrt((boundaries[i][0] - boundaries[i][2])**2+ (boundaries[i][1]- boundaries[i][3])**2)))
        #cv2.rectangle(frame, (boundaries[i][0], boundaries[i][1]), (boundaries[i][2], boundaries[i][3]), (0,0,255),2)
    
    #  Do some 'work'
    #randomsleeptime = random.uniform(0.4, 0.5)
    #time.sleep(randomsleeptime)
    
    socket_reid_rep.send_pyobj(dict(frame=frame, ts=time.time(), count=count, send_time=send_time, wc=wc))
'''
current_cam = 0
last_timestamp = time_now

detected = False
cleared = 0
while True:
    #  Wait for next request from client
    message = 0
    if current_cam ==0:
        message= socket_reid_req_0.recv_pyobj()

    elif current_cam ==1:
        message = socket_reid_req_1.recv_pyobj()
    else:
        message = socket_reid_req_2.recv_pyobj()

    time_before = time_now
    time_now = time.time()
    camnum = message['camnum']
    frame = message['frame']
    count = message['count']
    send_time = message['send_time']
    wc = message['wc']
    input_fps = message['input_fps']

    if wc == "EOF":
        print("[INFO] finishing at count {}".format(count))
        socket_reid_rep.send_pyobj(
            dict(camnum=camnum, frame=frame, ts=time.time(), count=count, send_time=send_time, wc="EOF"))
        continue
    if count == 0:
        print("[INFO] received request: cam{} count {}".format(camnum, count))
    else:
        print("[INFO] received request: cam{} count {},input fps {}, process fps : {}".format(camnum, count, input_fps,
                                                                                        1 / (time_now - time_before)))
    if str(type(frame)) == "<class 'NoneType'>":
        print("[INFO] Empty Frame!")
        continue

    # do object detection here.
    if cleared==0 and count>0:
        print("clearing queue")
        continue
    elif cleared==0 and count==0:
        cleared = 1
    elif send_time>=last_timestamp:
        detections, boundaries = detector.detect(frame)
        for i in range(len(detections)):
            print('[INFO] Object speed {} pixel per second'.format(
                math.sqrt((boundaries[i][0] - boundaries[i][2]) ** 2 + (boundaries[i][1] - boundaries[i][3]) ** 2)))
            #im = cv2.rectangle(frame, (boundaries[i][0], boundaries[i][1]), (boundaries[i][2], boundaries[i][3]), (0,0,255),2)
            #cv2.imwrite('frame{}-{}.png'.format(count,i), im)
        if len(detections) == 0:
            if detected==True:
                current_cam+=1
                current_cam=current_cam%total_num_cam
                last_timestamp = send_time
            detected=False
        else:
            detected = True
        cleared =1
    #  Do some 'work'
    # randomsleeptime = random.uniform(0.4, 0.5)
    # time.sleep(randomsleeptime)

    socket_reid_rep.send_pyobj(dict(camnum=camnum, frame=frame, ts=time.time(), count=count, send_time=send_time, wc=wc))