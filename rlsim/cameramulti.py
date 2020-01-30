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
import os

#vidpath = "/home/spencer/samplevideo/testedvids/edge-node"
#vidpath = "/home/spencer/samplevideo/testedvids/carla_6cam_" #shorter version (10sec)
#vidpath = "/home/spencer/samplevideo/testedvids/carla_4cam_" # lil longer version (40sec)
#vidpath = "/home/spencer/samplevideo/testedvids/random_6cam" # 6cam version (60sec)
#vidpath = "/home/spencer/samplevideo/testedvids/0.3likelihood_6cam_" # 6cam version (60sec)
#vidpath = "/home/spencer/samplevideo/testedvids_multipath/"
#vidpath = "/home/spencer/samplevideo/multipath_zonetozone/"
#vidpath = "/home/spencer/samplevideo/testedvids/real" # 6cam version (60sec)

#vidpath = vidpath = "/home/spencer/samplevideo/testedvids/carla_8cam_" 
class cam (object):
    def __init__ (self, id, iteration, like):
        self.id = id
        self.iteration = iteration
        self.likelihood = like
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
        self.lpos = ""
        #self.existing=None

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
        self.loadvid(id, iteration, like) 

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

        self.colors = pkl.load(open("pallete", "rb"))
        self.classes = self.load_classes ("data/coco.names")
        self.colors = [self.colors[1]]

    def load_classes(self, namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names       
        
    def procframe(self, id, idx):
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
        #print(id, dur_time,pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)
        
        self.trackers=[]
        positions=[]

        # if pre_score== '0' and pre_x1== '0' and pre_y1== '0' and pre_x2== '0' and pre_y2== '0' : # add new trackable object
        if pre_score!= 0 : # add new trackable object
            
            # find spot on matrix
            midx = ( pre_x1 + pre_x2 ) / 2
            midy = ( pre_y1 + pre_y2 ) / 2

            self.lpos=self.findinmatrix(midx, midy, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
            tracker.start_track(crgb, rect)
            self.trackers.append(tracker)

            answer = True
        else:
            if self.voidtimer == 0: # finding phase
                self.void = 0
                self.lpos = "xx" # initial position in matrix.
            else: # in btw cam
                self.void = int(time.time() - self.voidtimer)
                #print("when gone", self.void)
            answer = False
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
        self.existing = self.ct.checknumberofexisting() # 
        
        if self.existing: 
            self.voidtimer = time.time()
            # renew timer. 
            self.void = int(time.time() - stime)
            answer=True
        
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
                    prex, prey = self.ct.predict(objectID, 20)
                    #print("predicted obj loc x, y", prex, prey)
                    #cv2.circle(frame, (int(prex), int(prey)), 4, (0, 255,255),-1) # yellow: predicted location in 20 spots

            self.trobs[objectID] = to
        #cv2.imwrite(str(idx)+".jpg", frame)

        return answer, self.lpos
        
    def findinmatrix(self, x, y, w, h): # fixed size of matrix cell numbers: 10
        celldiv = 4
        cellwidth = w / celldiv 
        cellheight = h / celldiv
        tempx, tempy = 0, 0
        for i in range (0, celldiv):            
            if x > (i * cellwidth): 
                tempx = i
            elif x <= (i * cellwidth):
                break
                    
        for j in range (0, celldiv):
            if y > (j * cellheight):
                tempy = j
            elif y <= (j * cellheight):
                break
               
        return str(tempx)+str(tempy) # return the matrix location of the currently seen object

    def loadvid(self, id, iteration, like):
        # depending on the id of the cam number, load a diff vid.
        # vidpath = "/home/spencer/samplevideo/testedvids/"
        # likelihoodname = "likelihood_6cam_"
        # tmp  = "carla_6cam_"
        # self.vf = vidpath+str(like)+likelihoodname+str(iteration)+"/"+tmp+id+".avi"            
        # self.cap = cv2.VideoCapture(self.vf)
        # print(id, self.vf)
        
        #vidpath = "/home/spencer/samplevideo/testedvids_multipath/12cams_modified_trajectory"
        #vidpath = "/home/spencer/samplevideo/multipath_zonetozone/6cam_zone_"
        #vidpath = "/home/spencer/samplevideo/multipath_start4to/6cam_zone_"

        #likelihoodname = "12cams_modified_trajectory"
        tempvf = "6cam_zone_"+str(iteration)
        #list_of_files=os.listdir("/home/spencer/samplevideo/multipath_zonetozone/")
        #list_of_files=os.listdir("/home/spencer/samplevideo/multipath_start4to/")
        list_of_files=os.listdir("/home/spencer/samplevideo/multipath_start4to_150iter/")
        #list_of_files=os.listdir("/home/spencer/samplevideo/multipath_zonetozone_alldir/")

        print("iteration: ", iteration)
        for each_folder in list_of_files:
            
            if each_folder.startswith(tempvf): 
                #self.vf = "/home/spencer/samplevideo/multipath_zonetozone/"+each_folder+"/"+id+".avi"
                #self.vf = "/home/spencer/samplevideo/multipath_start4to/"+each_folder+"/"+id+".avi"
                self.vf = "/home/spencer/samplevideo/multipath_start4to_150iter/"+each_folder+"/"+id+".avi"
                #self.vf = "/home/spencer/samplevideo/multipath_zonetozone_alldir/"+each_folder+"/"+id+".avi"

        #self.vf = vidpath+str(iteration)+"/"+id+".avi"            
        
        self.cap = cv2.VideoCapture(self.vf)
        print("loading...", id, self.vf)


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
                        print(pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)      

            end_time = time.time()
            dur_time = end_time-start_time
            #print(dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2)
            return dur_time, pre_score, pre_class, pre_x1, pre_y1, pre_x2, pre_y2

        else: 
            end_time = time.time()
            dur_time = end_time-start_time
            return dur_time, 0, 0, 0, 0, 0, 0
