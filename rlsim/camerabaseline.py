#!/usr/bin/env python
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

#vidpath = "/home/spencer/samplevideo/testedvids/edge-node"
#vidpath = "/home/spencer/samplevideo/testedvids/carla_2cam_" #shorter version (10sec)
#vidpath = "/home/spencer/samplevideo/testedvids/carla_4cam_" # lil longer version (40sec)
#vidpath = "/home/spencer/samplevideo/testedvids/carla_6cam_" # lil longer version (40sec)
#vidpath = "/home/spencer/samplevideo/testedvids/random_6cam" # 6cam version (60sec)
vidpath = "/home/spencer/samplevideo/testedvids/0.7likelihood_6cam_" # 6cam version (60sec)
#vidpath = "/home/spencer/samplevideo/testedvids/random_6cam11/carla_6cam_"
class cam (object):
    def __init__ (self, id, iteration):
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
        self.width = 0
        self.height = 0
        
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
        self.loadvid(id, iteration) 

        # tracking related
        self.ct = centroidtracker.CentroidTracker(5, maxDistance = 20, queuesize=5)
        self.trobs = {}
        self.trackers = []

        # baseline eval
        self.TP=0
        self.TN=0
        self.FP=0
        self.FN=0

    def setupmodels(self, id):
        #prototxtpath = "/home/spencer/spencer/chameleon/resource/model_mSSD/MobileNetSSD_deploy.prototxt.txt"
        #modelpath = "/home/spencer/spencer/chameleon/resource/model_mSSD/MobileNetSSD_deploy.caffemodel"
        #self.net = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)    

        # https://discuss.pytorch.org/t/difference-device-error-when-i-use-multiple-gpus-with-creating-new-cuda-tensor/41676

        self.model = Darknet("cfg/yolov3.cfg")
        self.model.load_weights('yolov3.weights')
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

        self.colors = pkl.load(open("pallete", "rb"))
        self.classes = self.load_classes ("data/coco.names")
        self.colors = [self.colors[1]]

    def load_classes(self, namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names       
        
    
    def procframe(self, id, newact):
        #print("[incam] newact, ", newact)
        est = str(5)
        answer = "F"
        #print(type(newact))
        ret, frame = self.cap.read() # read next frame. 
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        crgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stime = time.time()
        if frame is None:
            print("no more frames, exiting")
            sys.exit(0)            
        elif frame is False:
            print("smth went wrong")
            sys.exit(0)
    
        dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2=self.detection_gpu_return(frame, id)
        #print(id, dur_time,pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)
        # do tracking here?
        self.trackers=[]
        positions=[]

        # if pre_score== '0' and pre_x1== '0' and pre_y1== '0' and pre_x2== '0' and pre_y2== '0' : # add new trackable object
        if pre_score!= 0 : # add new trackable object
            #print("in tracking", id, dur_time,pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)
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
                # 가장 최근 값으로 업데이트 
                self.void = int(time.time() - stime)
                #print("when exist", self.void)

            answer = True
                # calc void timer
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
                    #print("diry, dirx: ", dirY, dirX)
                    cv2.circle(frame, (int(dirX), int(dirY)), 4, (0, 0,255),-1) #cyan: current location based on 5 spots
                    to.centroids.append(centroid)

                    if not to.counted:
                        prex, prey = self.ct.predict(objectID, 10)
                        #print("predicted obj loc x, y", prex, prey)
                        cv2.circle(frame, (int(prex), int(prey)), 4, (0, 255,255),-1) # yellow: predicted location in 20 spots
                        if(self.check_boundary_dir(prex, prey)=="R"):
                            #print("we need to send msg to right")
                            p = self.ct.get_object_rect_by_id(objectID)

                            if (p[0]<=0 or p[1] <=0 or p[2]<=0 or p[3]<=0):
                                pass
                            else:
                                #est = (-1.8 * prex) + 8.15+1
                                
                                if str(self.id) == str(newact): # 7,7,3,1,3. 6,6,4,4,5
                                    if str(self.id)=="0":
                                        est = 6
                                    elif str(self.id)=="1":
                                        est = 6
                                    elif str(self.id)=="2":
                                        est = 4
                                    elif str(self.id)=="3":
                                        est = 4
                                    elif str(self.id)=="4":
                                        est = 5
                                    return str(est)


                self.trobs[objectID] = to
            #cv2.imwrite(str(idx)+".jpg", frame)
            
        else:
            if self.voidtimer == 0: # finding phase
                self.void = 0
            else: # in btw cam
                self.void = int(time.time() - self.voidtimer)
            #print("when gone", self.void)
            answer = False

        if newact == "find": # give T, F
            if answer == False:
                self.TN +=1 # fn --> TN, TN --> FN
            else:
                self.TP +=1
        elif str(newact) == str(self.id):
            if answer==False:
                self.FN+=1
            else:
                self.TP+=1
        elif str(newact) != str(self.id):
            if answer == False:
                self.TN+=1
            else:
                self.FP+=1
        elif newact == "SHUT":
            if answer == False:
                self.TN+=1
            else:
                self.FP+=1
        return answer

    def loadvid(self, id, iteration):
        # depending on the id of the cam number, load a diff vid.
        tmp  = "carla_6cam_"
        self.vf = vidpath+str(iteration)+"/"+tmp+id+".avi"
        self.cap = cv2.VideoCapture(self.vf)
        print(id, self.vf)
             
    def check_boundary_dir(self, prex, prey):
        #print(centroid)
        if(prey <= 0 and prex > 0 and prex <= self.width):
            return "U"
        elif(prex > self.width and prey > 0 and prey <= self.height):
            return "R"
        elif(prex <= 0 and prey > 0 and prey <=self.height):
            return "L"
        elif(prey > self.height and prex > 0 and prex <= self.width):
            return "D"
        else: 
            return "-"

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

        #detections = self.model(frame_tensor, self.cuda).cpu()
        detections = self.model(frame_tensor, self.CUDA, self.id).cpu()
        detections = process_result(detections, self.confidence, self.nms_thresh)
        
        if len(detections) != 0:
            detections = transform_result(detections, [frame], self.input_size)
#            for detection in detections:
            for idx, detection in enumerate(detections): #what happens if there are more than 1?
                if(self.classes[int(detection[-1])]=="person"):
                    if float(detection[6]) > self.confidence:
                        pre_score = (float(detection[6])) # prediction score
                        pre_class = (self.classes[int(detection[-1])]) # prediction class
                        pre_x1 = (int(detection[1])) # x1
                        pre_y1 = (int(detection[2])) # y1
                        pre_x2 = (int(detection[3])) # x2 
                        pre_y2 = (int(detection[4])) # y2 
                        #print(pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)      

            end_time = time.time()
            dur_time = end_time-start_time
            #print(dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)
            return dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2

        else: 
            end_time = time.time()
            dur_time = end_time-start_time
            return dur_time, 0, 0, 0, 0, 0, 0
