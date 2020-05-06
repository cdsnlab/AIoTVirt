#!/usr/bin/env python

#### trackes a person from video file.
#### 

import cv2
import sys
import os
import time
import numpy as np
from mydarknet import Darknet
import json
import torch
import pickle as pkl
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
from torch.autograd import Variable
import pandas as pd
import argparse
import datetime
import dlib
import centroidtracker
import trackableobject
import threading
import tqdm
import requests

def slacknoti(contentstr):
    webhook_url = "https://hooks.slack.com/services/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"
    payload = {"text": contentstr}
    requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

class cam (object):
    def __init__ (self, id, vidpath):
        self.id = id
        # cam setting related
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
        self.setupmodels(id)
        self.nms_thresh = 0.5 # decides the threshold for multiple edge points.
        self.confidence = 0.6 # if it goes under 0.6, the detecter sees smth that is not human

        # initialize video
        self.vidpath = vidpath
        self.loadvid(vidpath)
        
        self.ct = centroidtracker.CentroidTracker(3, maxDistance = 50, queuesize=10)
        self.trobs = {}
        self.trackers = []


    def setupmodels(self, id):

        # https://discuss.pytorch.org/t/difference-device-error-when-i-use-multiple-gpus-with-creating-new-cuda-tensor/41676

        self.model = Darknet("/home/spencer1/yolov3/yolov3.cfg")
        self.model.load_weights('/home/spencer1/yolov3/yolov3.weights')
        # self.model = Darknet("yolov3_tiny/yolov3_tiny.cfg")
        # self.model.load_weights('yolov3_tiny/yolov3_tiny.weights')
        if self.CUDA:
            
            if int(id) % 2 == 0:
                self.model.to("cuda:0")
                print("01_cuda set")
            else:
                self.model.to("cuda:1")
                print("02_cudaset")
            
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
            
    def procframe(self, id):
        centx, centy, prex, prey = -1, -1, -1, -1
        
        ret, frame = self.cap.read() # read next frame. 
        if frame is None:
            print("no more frames, exiting")
            sys.exit(0)            
        elif ret is False:
            print("smth went wrong")
            sys.exit(0)
        crgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2=self.detection_gpu_return(frame, id)
        pre_w = pre_x2-pre_x1
        pre_h = pre_y2-pre_y1
        centx = ( pre_x1 + pre_x2 ) / 2
        centy = ( pre_y1 + pre_y2 ) / 2
        # do tracking here?
        self.trackers=[]
        positions=[]

        if pre_score != -1:

            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
            tracker.start_track(crgb, rect)
            self.trackers.append(tracker)

        #print('making numbers...', )
        for tracker in self.trackers:
            #print("hmm")
            tracker.update(crgb)
            fpos = tracker.get_position()
            startX = int(fpos.left())
            startY = int(fpos.top())
            endX = int(fpos.right())
            endY = int(fpos.bottom())
            positions.append((startX, startY, endX, endY))

        objects = self.ct.update(positions)
        # for all objects in basket, predict movement direction. 
        for (objectID, centroid) in objects.items():
            #print(objectID, centroid)
            to = self.trobs.get(objectID, None)
            if to == None: # if there isn't any tracking object, itit one. 
                to = trackableobject.TrackableObject(objectID, centroid)
            else: 
                frame= cv2.circle(frame, (centroid[0], centroid[1]), 6, (255,255,0),-1) # BGR: CYAN
                y = [c[1] for c in to.centroids]
                x = [c[0] for c in to.centroids]
                #dirY = centroid[1] - np.mean(y)
                #dirX = centroid[0] - np.mean(x)
                #print("diry, dirx: ", dirY, dirX)
                #cv2.circle(frame, (int(dirX), int(dirY)), 4, (0, 0,255),-1) #BGR: red: current location based on 5 spots
                to.centroids.append(centroid)

                if not to.counted:
                    prex, prey = self.ct.predict(objectID, 3)
                    #print("predicted obj loc x, y", prex, prey)
                    frame=cv2.circle(frame, (int(prex), int(prey)), 4, (0, 255,255),-1) # BGR: yellow: predicted location in 20 spots
                    #return prex, prey

            self.trobs[objectID] = to
        #cv2.imwrite("herewego.png", frame)

        if pre_score != -1:
            #print("actually here")
            return (centx, centy, pre_w, pre_h)
        # elif self.ct.checknumberofexisting()==0:
        #     #print("nothing here")
        #     return (-1, -1, -1, -1)
        
        else:
            #print("virtually here")
            return (-1, -1, -1, -1)
        

    def loadvid(self, vp):
        # depending on the id of the cam number, load a diff vid.
        self.cap = cv2.VideoCapture(vp)
             

    def detection_gpu_return(self, frame, id):
        start_time = time.time()
        frame_tensor = cv_image2tensor(frame, self.input_size).unsqueeze(0)
        dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2 = -1, -1, -1, -1, -1, -1, -1
        frame_tensor = Variable(frame_tensor)

        if self.CUDA:
            frame_tensor = frame_tensor.to("cuda")
            if int(id) % 2 == 0:
                frame_tensor = frame_tensor.to("cuda:0")
            else:
                frame_tensor = frame_tensor.to("cuda:1")


        detections = self.model(frame_tensor, self.CUDA, self.id).cpu()
        start_detection = time.time()
        detections = process_result(detections, self.confidence, self.nms_thresh)
        end_detection = time.time()

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
            cv2.circle(frame, (int((pre_x1+pre_x2)/2), int((pre_y1+pre_y2)/2)), 4, (0, 255,255),-1) #             cv2.imshow("lets see", frame)
            cv2.imshow("wefw", frame)
            return dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2

        else: 
            end_time = time.time()
            dur_time = end_time-start_time
            return dur_time, -1, -1, -1, -1, -1, -1


def procvideo(video, camid):
    lst=[]
    camera = cam(str(camid), video)
    #print(camera.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame in range (0, int(camera.cap.get(cv2.CAP_PROP_FRAME_COUNT)-5)):
        centx, centy, prew, preh= camera.procframe(camera.id)
        # print(frame, x, y)

        tmplst = (centx, centy, prew, preh)
        lst.append(tmplst)
    #print(len(lst))
    return lst

def loop(args):

    fp = args.fp 
    for subdir, dirs, files in os.walk(fp):
        #if os.path.join(subdir, file) 
        columns = ['fn', 'cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7', 'cam8', 'cam9']
        #dfobj=pd.DataFrame(columns=columns)
        
        #print(subdir)
        lst=[]
        
        for file in files:
            if '.csv' in file:
                continue

            print (os.path.join(subdir,file))
            #print(file[-5])
            lst.extend(procvideo(os.path.join(subdir, file), str(file[0]))) #save to list or dictionary
            #lst.extend(procvideo(os.path.join(subdir, file), str(file[-5]))) #save to list or dictionary
            #lst.extend(procvideo(os.path.join(subdir, file), str(file[:-4]))) #save to list or dictionary
        #dfobj.append(lst)
        dfobj = pd.DataFrame(lst, columns=columns)
        print(os.path.join(subdir))
        print("creating csv: ", os.path.join(subdir))
        #dfobj.to_csv(os.path.join(subdir)+".csv")

def looper (args):
    fp = args.fp
    count=0
    start_time = time.time()
    #for subdir, dirs, files in tqdm (list(os.walk(fp))):
    for subdir, dirs, files in os.walk(fp):       
        print("creating csv in : {}".format(os.path.join(subdir)))
        dfobj=pd.DataFrame()
        for i, file in enumerate(files): # camera files
            lst=[]
            if '.csv' in file:
                pass
            else: 
                lst = procvideo(os.path.join(subdir, file), str(file[0]))
                dfobj[str(file[0])] = pd.Series(lst) # 
#                dfobj["Camera "+str(file[0])] = pd.Series(lst) # this may be a better option then a simple digit for the column

        
        dfobj.to_csv(os.path.join(subdir)+".csv")
        loop_time = time.time()
        slacknoti(" [INFO] {} more to go, elapsed time from start {} ".format((len(dirs)-count), loop_time-start_time))
        print(" [INFO] {} more to go, elapsed time from start {} ".format((len(dirs)-count), loop_time-start_time))
        count+=1
    


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description="welcome")
    argparser.add_argument(
        '--fp',
        metavar = 'f',
        #default = "/home/spencer/samplevideo/multi10zone_samplevideo/start_4_end_9_run_25/", # start_4_end_9_run_25
        default = "/home/spencer1/samplevideo/start1/",
        help='video folder location'
    )
    
    args = argparser.parse_args()
    # loop(args)
    looper(args)
