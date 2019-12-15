#!/usr/bin/env python
# -*- coding: utf-8 -*-
# referenced darknet from the following github.
# https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/blob/master/detect.py

import time
import cv2
import sys
import time
import torch
import threading
import numpy as np
from torch.autograd import Variable
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
from mydarknet import Darknet
import pickle as pkl
import dlib
import centroidtracker
import trackableobject
import requests
import json
from env1 import chamEnv
import argparse
import datetime


class cam (object):
    def __init__ (self, id, vidpath):
        self.id = id
        # cam setting related
        self.fr = None # framerate
        self.res = None # resolution
        self.algo = None # algorithm
        self.hot = None # handover time
        self.vf = None # opened video file
        self.cap = None # cap handle
        self.void = 0 # time when target is gone from view
        self.voidtmp = 0
        self.voidtimer = 0
        
        # NN related
        self.CUDA = torch.cuda.is_available()
        self.model = None
        self.input_size = None
        self.colors = None
        self.classes = None
        self.device = None
        self.setupmodels(id)
        self.nms_thresh = 0.5 # decides the threshold for multiple edge points.
        self.confidence = 0.6 

        # initialize video
        self.vidpath = vidpath
        self.loadvid(id)

        # tracking related
        self.ct = centroidtracker.CentroidTracker(10, maxDistance = 50, queuesize=10)
        self.trobs = {}
        self.trackers = []

    def setupmodels(self, id):
        #prototxtpath = "/home/spencer/spencer/chameleon/resource/model_mSSD/MobileNetSSD_deploy.prototxt.txt"
        #modelpath = "/home/spencer/spencer/chameleon/resource/model_mSSD/MobileNetSSD_deploy.caffemodel"
        #self.net = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)    

        # https://discuss.pytorch.org/t/difference-device-error-when-i-use-multiple-gpus-with-creating-new-cuda-tensor/41676

        self.model = Darknet("/home/spencer/mycfg/cfg/yolov3.cfg")
        self.model.load_weights('/home/spencer/mycfg/yolov3.weights')
        #self.cuda = torch.device(1)
        if self.CUDA:
            #self.model.to("cuda:0")
            #print("0_cuda set")
            if int(id) % 2 == 0:
                self.model.to("cuda:0")
                print("0_cuda set")
            else:
                self.model.to("cuda:1")
                print("1_cudaset")
        self.model.eval()
        self.setup_before_detection_gpu()
        print("setup complete")

    def setup_before_detection_gpu(self):
        self.input_size = [int(self.model.net_info['height']), int(self.model.net_info['width'])]
        # CANNOT MODIFY self.input_size yourself: self.input_size = [208, 208]

        self.colors = pkl.load(open("pallete", "rb"))
        self.classes = self.load_classes ("data/coco.names")
        self.colors = [self.colors[1]]

    def load_classes(self, namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names       
        
    
    def procframe(self, id, idx, newact):
        # which frame are we in.
        # print("processing frame id %d" % idx)

        answer = False
        ret, frame = self.cap.read() # read next frame. 
        crgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stime = time.time()
        if frame is None:
            print("no more frames, exiting")
            sys.exit(0)            
        elif frame is False:
            print("smth went wrong")
            sys.exit(0)
    
        dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2=self.detection_gpu_return(frame, id)
        # do tracking here?
        self.trackers=[]
        positions=[]

        # if pre_score== '0' and pre_x1== '0' and pre_y1== '0' and pre_x2== '0' and pre_y2== '0' : # add new trackable object
        if pre_score!= 0 : # add new trackable object
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
            tracker.start_track(crgb, rect)
            self.trackers.append(tracker)

            # keep on tracking 
            for tracker in self.trackers:
                tracker.update(crgb)
                fpos = tracker.get_position()
                startX = int(fpos.left())
                startY = int(fpos.top())
                endX = int(fpos.right())
                endY = int(fpos.bottom())
                positions.append((startX, startY, endX, endY))
            
            # update tracking objects positions 
            objects = self.ct.update(positions)
            self.existing = self.ct.checknumberofexisting()
            if self.existing: 
                self.voidtimer = time.time()
                self.void = int(time.time() - stime)

            # for all objects in basket, predict movement direction. 
            for (objectID, centroid) in objects.items():
                to = self.trobs.get(objectID, None)
                if to == None: # if there isn't any tracking object, itit one. 
                    to = trackableobject.TrackableObject(objectID, centroid)
                else: 
                    cv2.circle(frame, (centroid[0], centroid[1]), 6, (255,255,0),-1)
                    y = [c[1] for c in to.centroids]
                    x = [c[0] for c in to.centroids]
                    dirY = centroid[1] - np.mean(y)
                    dirX = centroid[0] - np.mean(x)
#                    print("diry, dirx: ", dirY, dirX)
                    cv2.circle(frame, (int(dirX), int(dirY)), 4, (0, 0,255),-1) #cyan: current location based on 5 spots
                    to.centroids.append(centroid)

                    if not to.counted:
                        prex, prey = self.ct.predict(objectID, 20)
#                        print("predicted obj loc x, y", prex, prey)
                        cv2.circle(frame, (int(prex), int(prey)), 4, (0, 255,255),-1) # yellow: predicted location in 20 spots

                self.trobs[objectID] = to
            answer = True
        else:
            if self.voidtimer == 0: # finding phase
                self.void = 0
            else: # in btw cam
                self.void = int(time.time() - self.voidtimer)
            answer = False
        return answer


    def loadvid(self, id):
        # depending on the id of the cam number, load a diff vid.
        self.vf = self.vidpath+id+".avi"
        self.cap = cv2.VideoCapture(self.vf)
        print(id, self.vf)
             
    

    def detection_gpu_return(self, frame, id):
        start_time = time.time()
        frame_tensor = cv_image2tensor(frame, self.input_size).unsqueeze(0)
        dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2 = 0,0,0,0,0,0,0
        frame_tensor = Variable(frame_tensor)

        if self.CUDA:
            # frame_tensor = frame_tensor.to("cuda:0")
            if int(id) % 2 == 0:
                frame_tensor = frame_tensor.to("cuda:0")
            else:
                frame_tensor = frame_tensor.to("cuda:1")


        detections = self.model(frame_tensor, self.CUDA, self.id).cpu()
        start_detection = time.time()
        detections = process_result(detections, self.confidence, self.nms_thresh)
        end_detection = time.time()
#        print("Elapsed detection time: ", end_detection-start_detection)

        if len(detections) != 0:
            detections = transform_result(detections, [frame], self.input_size)
            # for detection in detections:
            for idx, detection in enumerate(detections): #what happens if there are more than 1?
                if(self.classes[int(detection[-1])]=="person"):
                    if float(detection[6]) > self.confidence:
                        pre_score = (float(detection[6])) # prediction score
                        pre_class = (self.classes[int(detection[-1])]) # prediction class
                        pre_x1 = (int(detection[1])) # x1
                        pre_y1 = (int(detection[2])) # y1
                        pre_x2 = (int(detection[3])) # x2 
                        pre_y2 = (int(detection[4])) # y2 

            end_time = time.time()
            dur_time = end_time-start_time
            return dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2

        else: 
            end_time = time.time()
            dur_time = end_time-start_time
            return dur_time, 0, 0, 0, 0, 0, 0

        

### implementation ###
def slacknoti(contentstr):
   webhook_url = "https://hooks.slack.com/services/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"
   payload = {"text": contentstr}
   requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

def procvideo(current_video, id): 
    vidpath = current_video
    camera = cam(str(id), vidpath)
    time.sleep(2)
    frame = None
    prex={}
    prey={}

    for k in range(0, int(camera.cap.get(cv2.CAP_PROP_FRAME_COUNT))-2):
        #print(k)
        ret, frame = camera.cap.read() # read next frame. 

        #camera.trajectorypervideo(camera.id, k) # draw trajectory on each video, save as jpg.
        dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2=camera.detection_gpu_return(frame, id)
        prex[k]= (int((pre_x1+pre_x2)/2))
        prey[k]= (int((pre_y1+pre_y2)/2))
        #prex[k] = int((pre_x1+pre_x2)/2)
        #prey[k] = int((pre_y1+pre_y2)/2)
    
    for k in range(len(prex)):
        cv2.circle(frame, (int(prex[k]), int(prey[k])), 4, (0, 255,255),-1)
    cv2.imwrite(str(id)+".jpg", frame)

###################################################################################
slacknoti("[MEWTWO] spencer start simulation")
###################################################################################

# settings
path = "/home/spencer/samplevideo/multipath"
folders = ["6cam_multipath_trajectory0"]
files = ""

# if arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dirpath", default="noinput")
args = parser.parse_args()


# run program for specific folder
if args.dirpath != "noinput":
   dirpath = str(args.dirpath)
   if dirpath.endswith("/"):
      dirpath = dirpath[:len(dirpath)-1]

   #print("processing video sets in %s" % dirpath)
   detected_time = []  # list to hold detected time
   for j in range(6):
      current_video = dirpath + "/" + files
      print(current_video)
      slacknoti("[MEWTWO] spencer running:  processing  '%s%d'  in  '%s'" % (files, j, dirpath))
      procvideo(current_video, j)
   slacknoti("[MEWTWO] spencer end simulation")
   exit(0)


###################################################################################
slacknoti("[MEWTWO] hyoungjo end simulation")
###################################################################################









