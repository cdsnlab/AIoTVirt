import argparse
import configparser
import os
import sys
sys.path.insert(0, '../messaging')
from message_bus import MessageBus
from datetime import datetime
from util import process_result, load_images, resize_image, cv_image2tensor, transform_result
import base64
import random
import cv2
import numpy as np
import signal
import imutils
import psutil
import queue
import threading
import time
import torch
from torch.autograd import Variable
from darknet import Darknet
import pickle as pkl
import math
import ntplib
import trackableobject
import centroidtracker
import dlib

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class Controller(object):
    def __init__(self, name, port):
        self.msg_bus = MessageBus(name, port, 'controller')
        self.msg_bus.register_callback('join', self.handle_message)
        self.msg_bus.register_callback('img_e1-1', self.handle_message)
        self.msg_bus.register_callback('img_e1-2', self.handle_message)
        self.msg_bus.register_callback('img_e2', self.handle_message)
        self.msg_bus.register_callback('img_p', self.handle_message)
        self.msg_bus.register_callback('control_op', self.handle_message)
        self.model = None
        signal.signal(signal.SIGINT, self.signal_handler)
        self.logfile1 = None
        self.logfile2 = None
        self.logfile3 = None
        self.logfile4 = None
        self.dalgorithm = "yolo"
        self.starttime = 0.0
        self.endtime = 0.0
        self.cpu = None
        self.ram = None
        self.label_path = None
        self.classes = None
        self.cuda = None
        self.colors = None
        self.input_size = None
        self.confidence = None
        self.nms_thresh = None
        self.net = None # load caffe model
        self.framecnt = 0
        self.gettimegap()
        self.tr = None
        self.ts = None
        self.trackingscheme = None
        self.encode_param = None
        self.q_list = {} # this tracks which devices are sending frames.
        self.d_cam = {} # xxxxx copy version of dd_cam
        self.dd_cam = {} # this tells if a camera queue is empty or not
        self.exist = {} # this tells which device the tracking object is in
        self.d_list =[] # this tracks the name of sent devices. 
        self.numberofcameras = 0
        self.totalrecbytes = 0
        self.cur_tar_dev = None # tells which device has the target currently.

        self.imgq ={}
        self.timerq = {}
        self.framecntq ={}
        self.dev_nameq ={}
        self.typeq = {}

        # tracking related 
        self.ct = centroidtracker.CentroidTracker(50, maxDistance =50, queuesize = 10)
        self.frame_skips = None # how many frames should be skipped before detection 
        self.trackers = []
        self.trobs = {}
        self.boundary ={}
        self.objstatus ={}
        self.where={}
        self.framethr = 50
        self.sumframebytes = 0
        self.movingdelta = 0
        self.futuresteps = 0
        self.frame_skip = 10
        self.sumofframebytes = 0

    def gettimegap(self):
        starttime = datetime.now()
        ntp_response = ntplib.NTPClient().request('2.kr.pool.ntp.org', version=3)
        returntime = datetime.now()
        self.timegap = datetime.fromtimestamp(ntp_response.tx_time) - starttime - (returntime - starttime) / 2


    def signal_handler(self, sig, frame):
        self.msg_bus.ctx.destroy()
        self.logfile1.close()
        self.logfile2.close()
        self.logfile3.close()
        self.logfile4.close()
        print('closing logfile')
        torch.cuda.empty_cache()
        print('clearing cuda cache')
        sys.exit(0)

    def draw_bbox(self, imgs, bbox, colors, classes):
        img = imgs[int(bbox[0])]
        label = classes[int(bbox[-1])]
        p1 = tuple(bbox[1:3].int())
        p2 = tuple(bbox[3:5].int())
        color = random.choice(colors)
        cv2.rectangle(img, p1, p2, color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])
        cv2.rectangle(img, p3, p4, color, -1)
        cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)

    def handle_message(self, msg_dict):
        if msg_dict['type'] == 'join':
            self.handle_join(msg_dict)
        elif msg_dict['type'] == 'img_e1-1': # image raw data
            self.process_raw_tracking(msg_dict)
        elif msg_dict['type'] == 'img_e1-2': # image raw data
            self.process_raw_tracking(msg_dict)
        elif msg_dict['type'] == 'img_e2': # image raw data
            self.process_raw_tracking(msg_dict)
        elif msg_dict['type'] == 'img_p': # 
            self.process_raw_tracking(msg_dict)
        elif msg_dict['type'] == 'control_op':
            print("controller order")
        else:
            # Invalid type.
            pass

    def handle_join(self, msg_dict):
        node_table = self.msg_bus.node_table
        node_table.add_entry(msg_dict['device_name'], msg_dict['ip'], int(msg_dict['port']), msg_dict['location'], msg_dict['capability'])
        print('@@Table: ', self.msg_bus.node_table.table)

        # Send node_table
        device_list_json = {"type": "device_list", "devices": self.msg_bus.node_table.get_list_str()}
        for node_name in node_table.table.keys():
            node_info = node_table.table[node_name]
            self.msg_bus.send_message_json(node_info.ip, int(node_info.port), device_list_json)

    @threaded
    def process_raw_tracking(self, msg_dict):
#        print(' - tracking image')
        decstr = base64.b64decode(msg_dict['img_string'])
        imgarray = np.fromstring(decstr, dtype=np.uint8)
        decimg = cv2.imdecode(imgarray, cv2.IMREAD_COLOR)
        localNow = datetime.utcnow()+self.timegap

        # use this for transmission time btw device n edge server
        curTime = datetime.utcnow().strftime('%H:%M:%S.%f') # string forma
        curdatetime = datetime.strptime(curTime, '%H:%M:%S.%f')
        sentdatetime = datetime.strptime(msg_dict['time'], '%H:%M:%S.%f')
        simplecurtime = time.time()

        if(msg_dict['device_name'] in self.dd_cam.keys()): # depending on the sent device, add them to the corresonding queue.
            self.typeq[msg_dict['device_name']].put(str(msg_dict['type']))
            self.imgq[msg_dict['device_name']].put(decimg)
            self.timerq[msg_dict['device_name']].put(simplecurtime)
            self.framecntq[msg_dict['device_name']].put(str(msg_dict['framecnt']))
            self.dev_nameq[msg_dict['device_name']].put(str(msg_dict['device_name']))
            self.dd_cam[msg_dict['device_name']] = 'notempty'
        else:
            self.typeq[msg_dict['device_name']] = queue.Queue(2000)
            self.imgq[msg_dict['device_name']] = queue.Queue(2000)
            self.timerq[msg_dict['device_name']] = queue.Queue(2000)
            self.framecntq[msg_dict['device_name']] = queue.Queue(2000)
            self.dev_nameq[msg_dict['device_name']] = queue.Queue(2000)

            self.typeq[msg_dict['device_name']].put(str(msg_dict['type']))
            self.imgq[msg_dict['device_name']].put(decimg)
            self.timerq[msg_dict['device_name']].put(simplecurtime)
            self.framecntq[msg_dict['device_name']].put(str(msg_dict['framecnt']))
            self.dev_nameq[msg_dict['device_name']].put(str(msg_dict['device_name']))

            self.d_list.append(msg_dict['device_name'])
            self.dd_cam[msg_dict['device_name']] = 'notempty'
            self.numberofcameras+=1
        self.d_cam = {k : v for k, v in self.dd_cam.items()}
        self.totalrecbytes += sys.getsizeof(decimg)
        self.logfile4.write(str(self.totalrecbytes / 1000000) + "\n")

    def checkboundary(self, centroid, objectID):
        #print(centroid)
        if(centroid[0]<=self.framethr):
            if(centroid[1]<=self.framethr):
                self.boundary[objectID] = "TL"
            elif(centroid[1]>self.framethr and centroid[1] <= (self.height-self.framethr)):
                self.boundary[objectID] = "CL"
            elif(centroid[1]>(self.width-self.framethr)):
                self.boundary[objectID] = "BL"
        elif (centroid[0]>self.framethr and centroid[0] <= (self.width - self.framethr)):
            if(centroid[1] <= self.framethr):
                self.boundary[objectID] = "TC"
            elif(centroid[1]>self.framethr and centroid[1] <= (self.height-self.framethr)):
                self.boundary[objectID] = "CC"
            elif(centroid[1]>(self.width-self.framethr)):
                self.boundary[objectID] = "BC"
        elif (centroid[0] > (self.width - self.framethr)):
            if(centroid[1] <= self.framethr):
                self.boundary[objectID] = "TR"
            elif(centroid[1]>self.framethr and centroid[1] <= (self.height-self.framethr)):
                self.boundary[objectID] = "CR"
            elif(centroid[1]>(self.width-self.framethr)):
                self.boundary[objectID] = "BR"

    def checkboundary_dir(self, prex, prey):
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
        
    def checkdir(self, dirX, dirY, objectID): # this -2, 2 must also be adjusted depending on the tracked objects
        #print(dirX, dirY, objectID)
        if dirY <= -(self.movingdelta):
            if dirX <= -(self.movingdelta):
                self.objstatus[objectID] = "NW"
            elif dirX <= (self.movingdelta) and dirX > -(self.movingdelta):
                self.objstatus[objectID] = "N"
            elif dirX > (self.movingdelta):
                self.objstatus[objectID] = "NE"
                            
        elif dirY <= (self.movingdelta) and dirY > -(self.movingdelta):
            if dirX <= -(self.movingdelta):
                self.objstatus[objectID] = "W"
            elif dirX <= (self.movingdelta) and dirX > -(self.movingdelta):
                self.objstatus[objectID] = "-"
            elif dirX > (self.movingdelta):
                self.objstatus[objectID] = "E"

        elif dirY > (self.movingdelta):
            if dirX <= -(self.movingdelta):
                self.objstatus[objectID] = "SW"
            elif dirX <= (self.movingdelta) and dirX > -(self.movingdelta):
                self.objstatus[objectID] = "S"
            elif dirX > (self.movingdelta):
                self.objstatus[objectID] = "SE"
        #print (self.objstatus[objectID])

    def checkhandoff(self, objectID):
        if(self.boundary[objectID] == "TL" or self.boundary[objectID] == "TC" or self.boundary[objectID] == "TR"):
            if (self.objstatus[objectID]=="NW" or self.objstatus[objectID]=="N" or self.objstatus[objectID]=="NE"):
                # do hand off request to next cam
                print ("need hand off to top cam.")
                return "TOP"
        elif(self.boundary[objectID] == "TR" or self.boundary[objectID] == "CR" or self.boundary[objectID] == "BR"):
            if (self.objstatus[objectID]=="NE" or self.objstatus[objectID]=="E" or self.objstatus[objectID]=="SE"):
                # do hand off request to next cam
                print ("need hand off to right cam.")
                return "RIGHT"
        elif(self.boundary[objectID] == "BL" or self.boundary[objectID] == "BC" or self.boundary[objectID] == "BR"):
            if (self.objstatus[objectID]=="SW" or self.objstatus[objectID]=="S" or self.objstatus[objectID]=="SE"):
                # do hand off request to next cam
                print ("need hand off to bottom cam.")
                return "BOTTOM"
        elif(self.boundary[objectID] == "TL" or self.boundary[objectID] == "CL" or self.boundary[objectID] == "BL"):
            if (self.objstatus[objectID]=="NW" or self.objstatus[objectID]=="W" or self.objstatus[objectID]=="SW"):
                # do hand off request to next cam
                print ("need hand off to left cam.")
                return "LEFT"
        else:
            return "CALM"

    def isitempty(self):
        full = 0
        for key, value in self.d_cam.items():
#            print(key, value)
            if (value == 'empty'):
                full +=1
        if full == self.numberofcameras: # if this matches, it means its not empty
            return True 
        else:
            return False

    @threaded
    def e1_1_proc_dequeue(self):
        framecnt = 0
        emptycount = 0
        endcnt =0 # if idle for 2 minutes, save and quit.
        frame_start_time = time.time()
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),99]
        while (True):
            print(self.isitempty())    
            if(self.isitempty()): # yes its empty...
                print('nothing in all q, sleeping..')
                time.sleep(0.5)
                endcnt += 1

            else: # not all q are empty... loop all queues until the target is found. 
                while(self.cur_tar_dev==None):
                    for i in self.d_cam.keys():
                        if(self.imgq[i].empty()): # if i'th queue is empty, skip this queue stak. check other camera queue stack.
                            self.d_cam[i]='empty'
                            print("[finding..] "+ i +" q is empty, moving on to next q") 
                            time.sleep(0.001) 
                            continue
                        else:
                            print("[finding..] ", i, self.dev_nameq[i].qsize(), self.framecntq[i].qsize(), self.imgq[i].qsize(), self.typeq[i].qsize(), self.timerq[i].qsize())
                            # it does not skip, u just didn't catch if its not human or low threshold... dumb f... 
                            self.d_cam[i] = 'notempty'
                            cdevice_name = self.dev_nameq[i].get()
                            ccounter = int(self.framecntq[i].get())
                            cframe = self.imgq[i].get()
                            ctype = self.typeq[i].get()
                            ctimer = self.timerq[i].get()
                            s = time.time()
                            if self.cuda:
                                width = cframe.shape[0]
                                height = cframe.shape[1]
                                crgb = cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB)
    
                                frame_tensor = cv_image2tensor(cframe, self.input_size).unsqueeze(0)
                                frame_tensor = Variable(frame_tensor)

                                frame_tensor = frame_tensor.cuda()

                                detections = self.model(frame_tensor, self.cuda).cpu()
                                detections = process_result(detections, self.confidence, self.nms_thresh)
                            
                                if len(detections) != 0:
                                    detections = transform_result(detections, [cframe], self.input_size)
                                    #for detection in detections:
                                    for idx, detection in enumerate(detections):
                                        if (self.classes[int(detection[-1])]=="person"):
                                            if float(detection[6]) > self.confidence:
                                                print("-[searching phase] FOUND person on device: ",ccounter, cdevice_name) # well u r supposed to match the person detect with the template, but for the time being we assume its a person. # sangwoo is on it thou
                                                pre_score = (float(detection[6])) # prediction score
                                                pre_class = (self.classes[int(detection[-1])]) # prediction class
                                                pre_x1 = (int(detection[1])) # x1
                                                pre_y1 = (int(detection[2])) # y1
                                                pre_x2 = (int(detection[3])) # x2 
                                                pre_y2 = (int(detection[4])) # y2

                                                #self.draw_bbox([frame], detection, self.colors, self.classes)
                                                tracker = dlib.correlation_tracker()
                                                rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
                                                tracker.start_track(crgb, rect)
                                                self.trackers.append(tracker)
                                                self.cur_tar_dev = cdevice_name
                                                break
                                            else:
                                                print("[finding..] low confidence person", ccounter, cdevice_name)
                                        else: 
                                            print("[finding..] not a person", ccounter, cdevice_name)

                                else:
                                    print("[finding..] nothing detected: ", ccounter, cdevice_name)
                            time_now = time.time()
                            inf_time = time_now - s
                            if cdevice_name == "camera01":
                                if self.cur_tar_dev == "camera01": # if this device found the target
                                    self.logfile1.write(str(ccounter) + "\t" + str(time_now - ctimer) + "\t" + "1" + "\t" + str(inf_time)+"\n")
                                else:
                                    self.logfile1.write(str(ccounter) + "\t" + str(time_now - ctimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")
                            elif cdevice_name == "camera02":
                                if self.cur_tar_dev == "camera02":
                                    self.logfile2.write(str(ccounter) + "\t" + str(time_now - ctimer) + "\t" + "1" + "\t" + str(inf_time)+"\n")
                                else:
                                    self.logfile2.write(str(ccounter) + "\t" + str(time_now - ctimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")

                        if self.isitempty(): # won't fall into unless there is a pass code above...
                            print('[finding..] still at search phase, all q empty, go back to waiting phase')
                            self.cur_tar_dev = 'wow' # a random txt to escape this loop
                        
                            break

                for i in self.d_cam.keys(): # target found phase.

                    if self.isitempty():
                        print('[target found phase..] break from searching phase, all q empty, go back to waiting phase')
                        self.d_cam[i] = 'empty' 
                        break
                    if(self.imgq[i].empty()): # if i'th queue is empty, skip this queue stak. check other camera queue stack.
                        self.d_cam[i]='empty'
                        print("[target found phase....] "+ i +" q is empty, moving on to next q") 
                        continue
                    else: # there is smth here 
                        print("[target_found_phase..] ", i, self.dev_nameq[i].qsize(), self.framecntq[i].qsize(), self.imgq[i].qsize(), self.typeq[i].qsize(), self.timerq[i].qsize())
                        self.d_cam[i] = 'notempty' 
                        ftype = self.typeq[i].get()
                        fdevice_name = self.dev_nameq[i].get()
                        fcounter = self.framecntq[i].get()
                        ftimer = self.timerq[i].get()
                        fframe = self.imgq[i].get()
                        rgb = cv2.cvtColor(fframe, cv2.COLOR_BGR2RGB)
                        fpositions=[]
                        fs = time.time()
                        if fdevice_name == self.cur_tar_dev:
                            if(int(fcounter) % self.frame_skips == 0):
                                self.trackers=[]
                                if self.cuda:
                                    width = fframe.shape[0]
                                    height = fframe.shape[1]

                                    frame_tensor = cv_image2tensor(fframe, self.input_size).unsqueeze(0)
                                    frame_tensor = Variable(frame_tensor)

                                    if self.cuda:
                                        frame_tensor = frame_tensor.cuda()

                                    detections = self.model(frame_tensor, self.cuda).cpu()
                                    detections = process_result(detections, self.confidence, self.nms_thresh)
                                    if len(detections) != 0:
                                        detections = transform_result(detections, [cframe], self.input_size)
                                    #for detection in detections:
                                        for idx, detection in enumerate(detections):
                                            if (self.classes[int(detection[-1])]=="person"):
                                                if float(detection[6]) > self.confidence:
                                                    print("-- [target found phase ] person on device: ", fdevice_name) # well u r supposed to match the person detect with the template, but for the time being we assume its a person. # sangwoo is on it thou
                                                    pre_score = (float(detection[6])) # prediction score
                                                    pre_class = (self.classes[int(detection[-1])]) # prediction class
                                                    pre_x1 = (int(detection[1])) # x1
                                                    pre_y1 = (int(detection[2])) # y1
                                                    pre_x2 = (int(detection[3])) # x2 
                                                    pre_y2 = (int(detection[4])) # y2

                                                #self.draw_bbox([frame], detection, self.colors, self.classes)
                                                    tracker = dlib.correlation_tracker()
                                                    rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
                                                    tracker.start_track(rgb, rect)
                                                    self.trackers.append(tracker)
                                                    self.cur_tar_dev = fdevice_name
                            else:
                                for tracker in self.trackers: 
                                    tracker.update(rgb)
                                    fpos = tracker.get_position()
                                    startX = int(fpos.left())
                                    startY = int(fpos.top())
                                    endX = int(fpos.right())
                                    endY = int(fpos.bottom())
                                    fpositions.append((startX, startY, endX, endY))

                            objects = self.ct.update(fpositions)
                            # track well and if u miss out, go back to search phase
                            self.exist[i] = self.ct.checknumberofexisting()
                            print(fdevice_name, self.exist[i])
#                            if self.ct.checknumberofexisting():
                            time_now = time.time()
                            inf_time = time_now - fs
                            if self.exist[i] and self.cur_tar_dev:
                                
                                print("[target tracking phase ] donno but still here: ", fdevice_name, fcounter, self.exist[i])
                                if fdevice_name == 'camera01':
                                    self.logfile1.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "1" + "\t" + str(inf_time)+"\n")
                                else:
                                    self.logfile2.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "1" + "\t" + str(inf_time)+"\n")
                            else: # break and find it now , but what if not lost ?
                                print("target lost--------------")
                                if fdevice_name == 'camera01':
                                    self.logfile1.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")
                                else:
                                    self.logfile2.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")
                                self.cur_tar_dev = None
#                                emptycount= 0
                                break

                        else: # not cur_tar_dev, drop frames but log
                            time_now = time.time()
                            inf_time = 0
                            if fdevice_name == "camera01":
                                if self.cur_tar_dev == "camera02":
                                    self.logfile1.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")
                            elif fdevice_name == "camera02":
                                if self.cur_tar_dev == "camera01":
                                    self.logfile2.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")

                            print("dropping frames from: ", fdevice_name)
                            pass

    @threaded
    def e1_2_proc_dequeue(self):
        framecnt = 0
        emptycount = 0
        endcnt =0 # if idle for 2 minutes, save and quit.
        frame_start_time = time.time()
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),99]
        node_table = self.msg_bus.node_table
        while (True):
            
            for i in self.d_cam.keys():
                print(i+" "+ self.d_cam[i])
                if (self.imgq[i].empty()):
                    self.d_cam[i] = 'empty'
            # check if all d_cams are empty
            for i in self.d_cam.keys():
                if self.d_cam[i] == 'empty':
                    emptycount += 1

            if emptycount == self.numberofcameras: # all q are empty.. loop around until one is filled up.
                if(endcnt >= 600):
                    self.logfile1.close()
                    self.logfile2.close()
                    print("byebye")
                    sys.exit(0)
                    emptycount=0
                    break
                else:
                    print("[q all empty] waiting..")
                    # send message to all cam devs to quit or start sending 
                    for node_name in node_table.table.keys():
                        node_info = node_table.table[node_name]
                        op_json = {"type": "control_op", "onoff": "True"}
                        print("[q all empty] telling "+node_info.device_name+" to start sending")
                        self.msg_bus.send_message_json(node_info.ip, int(node_info.port), op_json) 

                    time.sleep(0.5)
                    endcnt += 1
                    emptycount =0

            else: # not all q are empty... loop all queues until the target is found. 
                while(self.cur_tar_dev==None):
                    for i in self.d_cam.keys():
                        if(self.imgq[i].empty()): # if i'th queue is empty, skip this queue stak. check other camera queue stack.
                            self.d_cam[i]='empty'
                            emptycount+=1
                            print("[finding..] "+ i +" q is empty, moving on to next q") 
                            # continue
                            pass
                        else:
                            print(self.dev_nameq[i].qsize(), self.framecntq[i].qsize(), self.imgq[i].qsize(), self.typeq[i].qsize(), self.timerq[i].qsize())
                            s = time.time()
                            emptycount =0
                            cdevice_name = self.dev_nameq[i].get()
                            ccounter = int(self.framecntq[i].get())
                            cframe = self.imgq[i].get()
                            ctype = self.typeq[i].get()
                            ctimer = self.timerq[i].get()
                            if self.cuda:

                                width = cframe.shape[0]
                                height = cframe.shape[1]
                                crgb = cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB)
    
                                frame_tensor = cv_image2tensor(cframe, self.input_size).unsqueeze(0)
                                frame_tensor = Variable(frame_tensor)

                                frame_tensor = frame_tensor.cuda()

                                detections = self.model(frame_tensor, self.cuda).cpu()
                                detections = process_result(detections, self.confidence, self.nms_thresh)
                            
#print(self.dev_nameq[i].qsize(), self.framecntq[i].qsize(), self.imgq[i].qsize(), self.typeq[i].qsize(), self.timerq[i].qsize(), len(detections))
                                if len(detections) != 0:
                                    detections = transform_result(detections, [cframe], self.input_size)
                                    #for detection in detections:
                                    for idx, detection in enumerate(detections):
                                        if (self.classes[int(detection[-1])]=="person"):
                                            if float(detection[6]) > self.confidence:
                                                print("[finding..] found person on device: ",ccounter, cdevice_name) # well u r supposed to match the person detect with the template, but for the time being we assume its a person. # sangwoo is on it thou
                                                # send message to all cam devs to quit or start sending 
                                                for node_name in node_table.table.keys():
                                                    node_info = node_table.table[node_name]
                                                    if node_info.device_name == cdevice_name:
                                                        op_json = {"type": "control_op", "onoff": "True"}
                                                        print("[w] telling "+cdevice_name+" to start sending")
                                                    else: # if its the target is not there, signal stop  
                                                        op_json = {"type": "control_op", "onoff": "False"}
                                                        print("[w] telling "+cdevice_name+" to stop sending")
                                                    self.msg_bus.send_message_json(node_info.ip, int(node_info.port), op_json) 
                                                pre_score = (float(detection[6])) # prediction score
                                                pre_class = (self.classes[int(detection[-1])]) # prediction class
                                                pre_x1 = (int(detection[1])) # x1
                                                pre_y1 = (int(detection[2])) # y1
                                                pre_x2 = (int(detection[3])) # x2 
                                                pre_y2 = (int(detection[4])) # y2

                                                #self.draw_bbox([frame], detection, self.colors, self.classes)
                                                tracker = dlib.correlation_tracker()
                                                rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
                                                tracker.start_track(crgb, rect)
                                                self.trackers.append(tracker)
                                                self.cur_tar_dev = cdevice_name
                                                break
                                            else:
                                                print("[finding..] low confidence person", ccounter, cdevice_name)
                                        else: 
                                            print("[finding..] not a person", ccounter, cdevice_name)

                                else:
                                    print("[finding..] nothing detected: ", ccounter, cdevice_name)
                                    print(self.imgq[i].empty()) # true if empty
                            e = time.time()
                            print ("time for inf: ", e - s)
                            print ("decay time: ", e - ctimer)
                            print("log here")
                             

                    if emptycount == self.numberofcameras:
                        print('[finding..] still at search phase, all q empty, go back to waiting phase')
                        break

                for i in self.d_cam.keys():

                    if emptycount == self.numberofcameras:
                        print('[found..] break from searching phase, all q empty, go back to waiting phase')
                        emptycount =0
                        self.cur_tar_dev=None
                        break
                    else: # there is smth here 
                        emptycount =0
                        ftype = self.typeq[i].get()
                        fdevice_name = self.dev_nameq[i].get()
                        fcounter = self.framecntq[i].get()
                        ftimer = self.timerq[i].get()
                        fframe = self.imgq[i].get()
                        rgb = cv2.cvtColor(fframe, cv2.COLOR_BGR2RGB)
                        fpositions=[]
                        if fdevice_name == self.cur_tar_dev:
                            if(int(fcounter) % self.frame_skips == 0):
                                self.trackers=[]
                                if self.cuda:
                                    width = fframe.shape[0]
                                    height = fframe.shape[1]

                                    frame_tensor = cv_image2tensor(fframe, self.input_size).unsqueeze(0)
                                    frame_tensor = Variable(frame_tensor)

                                    if self.cuda:
                                        frame_tensor = frame_tensor.cuda()

                                    detections = self.model(frame_tensor, self.cuda).cpu()
                                    detections = process_result(detections, self.confidence, self.nms_thresh)
                                    if len(detections) != 0:
                                        detections = transform_result(detections, [cframe], self.input_size)
                                    #for detection in detections:
                                        for idx, detection in enumerate(detections):
                                            if (self.classes[int(detection[-1])]=="person"):
                                                if float(detection[6]) > self.confidence:
                                                    print("[found..] person on device: ", cdevice_name) # well u r supposed to match the person detect with the template, but for the time being we assume its a person. # sangwoo is on it thou
                                                    pre_score = (float(detection[6])) # prediction score
                                                    pre_class = (self.classes[int(detection[-1])]) # prediction class
                                                    pre_x1 = (int(detection[1])) # x1
                                                    pre_y1 = (int(detection[2])) # y1
                                                    pre_x2 = (int(detection[3])) # x2 
                                                    pre_y2 = (int(detection[4])) # y2

                                                #self.draw_bbox([frame], detection, self.colors, self.classes)
                                                    tracker = dlib.correlation_tracker()
                                                    rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
                                                    tracker.start_track(rgb, rect)
                                                    self.trackers.append(tracker)
                                                    self.cur_tar_dev = cdevice_name
                            else:
                                for tracker in self.trackers: 
                                    tracker.update(rgb)
                                    fpos = tracker.get_position()
                                    startX = int(fpos.left())
                                    startY = int(fpos.top())
                                    endX = int(fpos.right())
                                    endY = int(fpos.bottom())
                                    fpositions.append((startX, startY, endX, endY))

                            objects = self.ct.update(fpositions)
                            # track well and if u miss out, go back to search phase
                            if self.ct.checknumberofexisting():
                                print("[found..] donno but still here: ", fdevice_name)
                                self.sumframebytes+=sys.getsizeof(fframe)
                            else: # break and find it now 
                                self.cur_tar_dev = None
                                break
                        else:
                            continue # drop other queues
                        time_now = time.time()
                        inf_time = time_now - s
                        if cdevice_name == "camera01":
                            if self.cur_tar_dev != None:
                                self.logfile1.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "1" + "\t" + str(inf_time)+"\n")
                            else:
                                self.logfile1.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "0" + "\t" + str(inf_time))+"\n"
                        elif cdevice_name == "camera02":
                            if self.cur_tar_dev != None:
                                self.logfile2.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "1" + "\t" + str(inf_time)+"\n")
                            else:
                                self.logfile2.write(str(fcounter) + "\t" + str(time_now - ftimer) + "\t" + "0" + "\t" + str(inf_time)+"\n")


   
    @threaded
    def p_proc_dequeue(self):
        framecnt = 0
        endcnt =0 # if idle for 2 minutes, save and quit.
        frame_start_time = time.time()
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),99]
        while (True):
             if (self.imgq.empty()):
                if(endcnt >= 120):
                    self.logfile1.close()
                    self.logfile2.close()
                    print("byebye")
                    sys.exit(0)
                else:
                    print('nothing in q, sleeping..')
                    time.sleep(1)
                    endcnt += 1

             else:
                device_name = self.dev_nameq.get()
                counter = int(self.framecntq.get())
                frame = self.imgq.get()
                if self.cuda:
                    if self.typeq.get() == 'img': # just detection, nothing else
                        self.detection_gpu(self.model, frame, cnt)

                    elif self.typeq.get() == 'img_tracking':# for tracking
                        self.width = frame.shape[0]
                        self.height = frame.shape[1]
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        positions = []
                        if int(counter) % self.frame_skip == 0:
                            self.trackers = []
                            frame_tensor = cv_image2tensor(frame, self.input_size).unsqueeze(0)
                            frame_tensor = Variable(frame_tensor)

                            if self.cuda:
                                frame_tensor = frame_tensor.cuda()

                            detections = self.model(frame_tensor, self.cuda).cpu()
                            detections = process_result(detections, self.confidence, self.nms_thresh)
        
                            if len(detections) != 0:
                                detections = transform_result(detections, [frame], self.input_size)
                            #for detection in detections:
                                for idx, detection in enumerate(detections):
                                    if (self.classes[int(detection[-1])]=="person"):
                                        if float(detection[6]) > self.confidence:
                                            print("foound!")
                                            pre_score = (float(detection[6])) # prediction score
                                            pre_class = (self.classes[int(detection[-1])]) # prediction class
                                            pre_x1 = (int(detection[1])) # x1
                                            pre_y1 = (int(detection[2])) # y1
                                            pre_x2 = (int(detection[3])) # x2 
                                            pre_y2 = (int(detection[4])) # y2

                                            #self.draw_bbox([frame], detection, self.colors, self.classes)
                                            tracker = dlib.correlation_tracker()
                                            rect = dlib.rectangle(pre_x1, pre_y1, pre_x2, pre_y2)
                                            tracker.start_track(rgb, rect)
                                            self.trackers.append(tracker)
                        else: 
                            for tracker in self.trackers: 
                                tracker.update(rgb)
                                pos = tracker.get_position()
                                startX = int(pos.left())
                                startY = int(pos.top())
                                endX = int(pos.right())
                                endY = int(pos.bottom())
                                positions.append((startX, startY, endX, endY))

                        objects = self.ct.update(positions)

                        for (objectID, centroid) in objects.items():
                            to = self.trobs.get(objectID, None)
                            #self.ct.getall()
                            if (self.ct.checknumberofexisting()):
                                self.sumframebytes+=sys.getsizeof(frame)
                            #self.ct.predict(objectID, 30)

                            if to == None:
                                to = trackableobject.TrackableObject(objectID, centroid)
                            else:

                                cv2.circle(frame, (centroid[0],centroid[1]),4,(255,255,255),-1)
                                y = [c[1] for c in to.centroids]
                                x = [c[0] for c in to.centroids]
                                dirY = centroid[1] - np.mean(y)
                                dirX = centroid[0] - np.mean(x)
                                to.centroids.append(centroid)
                    
                                if not to.counted:
                                    if (self.ts == "dr"):
                                        prex, prey = self.ct.predict(objectID, 30)
                                        print("predicted obj movement..x,y: ", prex, prey)
                                        if(self.checkboundary_dir(prex, prey)=="R"):
                                            print("we need to send msg to right")
                                            p = self.ct.get_object_rect_by_id(objectID) # x1, y1, x2, y2
                                            if (p[0]<=0 or p[1] <= 0 or p[2] <= 0 or p[3] <=0):
                                                pass
                                            else:
                                                cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255,0,0),2)
                                                croppedimg = frame[p[1]:p[3], p[0]:p[2]]
                                                jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, self.encode_param, device_name, self.timegap)
                                                #self.msg_bus.send_message_str(self.center_device_ip_int, self.center_device_port, jsonified_data)

                                        # sned mesg
                                        elif(self.checkboundary_dir(prex, prey)=="L"):
                                            print("we need to send msg to left")
                                            p = self.ct.get_object_rect_by_id(objectID) # x1, y1, x2, y2
                                            if (p[0]<=0 or p[1] <= 0 or p[2] <= 0 or p[3] <=0):
                                                pass
                                            else:
                                
                                                cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255,0,0),2)
                                                croppedimg = frame[p[1]:p[3], p[0]:p[2]]
                                                print((p[0], p[1]), (p[2], p[3]))
                                                #cv2.imwrite(str(counter)+".jpg", croppedimg)
                                                jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, self.encode_param, device_name, self.timegap)
                                                #self.msg_bus.send_message_str(self.left_device_ip_int, self.left_device_port, jsonified_data)

                                        elif(self.checkboundary_dir(prex, prey)=="D"):
                                            print("we need to send msg to down")
                                        elif(self.checkboundary_dir(prex, prey)=="U"):
                                            print("we need to send msg to up")

                                    elif (self.ts == "bc"):
                                        self.checkboundary(centroid, objectID)
                                        #print("loc of object in frame: ", objectID, self.boundary[objectID])

                                        self.checkdir(dirX, dirY, objectID)
                                        #print("moving direction of the object: ", objectID, self.objstatus[objectID])

                                        self.where[objectID] = self.checkhandoff(objectID)
                                        # send hand off msg here
                                        if(self.where[objectID] == "RIGHT"):
                                            print("we need to send msg to right")
                                            #print("just throw in the cropped img template")
                                            #print("upon receiving the cropped img, do the template matching & add to tracking dlib queue")
                                            p = self.ct.get_object_rect_by_id(objectID) # x1, y1, x2, y2
                                            #t = self.ct.objects[objectID]  #centroid x, y
                                            #print("type p: ", type(p)) # rect
                                            #print("t: ", t) # centroid
                                            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255,0,0),2)
                                            #croppedimg = frame[y1:y2, x1: x2]
                                            croppedimg = frame[p[1]:p[3], p[0]:p[2]]
                                            jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, self.encode_param, device_name, self.timegap)
                                            self.msg_bus.send_message_str(self.center_device_ip_int, self.center_device_port, jsonified_data)

                            
                                        elif (self.where[objectID]== "LEFT"):
                                            print("we need to send msg to left")
                                            p = self.ct.get_object_rect_by_id(objectID) # x1, y1, x2, y2
                                            #t = self.ct.objects[objectID]  #centroid x, y
                                            #print("type p: ", type(p)) # rect
                                            #print("t: ", t) # centroid
                                            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255,0,0),2)
                                            #croppedimg = frame[y1:y2, x1: x2]
                                            croppedimg = frame[p[1]:p[3], p[0]:p[2]]
                                            jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, self.encode_param, device_name, self.timegap)
                                            #jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, encode_param, device_name, self.timegap)
                                            self.msg_bus.send_message_str(self.right_device_ip_int, self.right_device_port, jsonified_data)
                                        elif (self.where[objectID]== "TOP"):
                                            print("we need to send msg to top")
                                        elif (self.where[objectID]== "BOTTOM"):
                                            print("we need to send msg to bottom")
                                        else:
                                            print("nothing is happinging")

                            self.trobs[objectID] = to
                            text = "ID {}".format(objectID) +" "+ str(centroid[0]) +" "+ str(centroid[1])
                            cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2) # ID of the object 
                            
                        sendablecnt = 0
#                        cv2.imwrite('frame'+str(counter)+'.jpg', frame)
                        if self.display == "on":
                            cv2.imshow('NCS live inference', frame)
                        if(cv2.waitKey(3) & 0xFF == ord('q')):
                            break

                    elif self.typeq.get() == 'img_tracking_check':# dummy here
                        thing = self.detection_gpu_return(self.model, frame, cnt)
                else: # no GPU enabled
                    if self.typeq.get() == 'img':
                        self.detection(frame)
                    elif self.typeq.get() == 'img_tracking':
                        print ("NOT IMP YET")
                        # self.detection_tracking (self.model, self.imgq.get(), cnt)
                    elif self.typeq.get() == 'img_tracking_check':
                        print ("NOT IMP YET")
                        # thing = self.detection_gpu_return(self.model, self.imgq.get(), cnt)
                curT = datetime.utcnow().strftime('%H:%M:%S.%f') # string format
                decay = datetime.strptime(curT, '%H:%M:%S.%f') - self.timerq.get()

                frame_end_time = time.time()
                cumlative_fps = int(counter) / (frame_end_time - frame_start_time)
                print("estimated dequeue speed: ", str(cumlative_fps)) # i don't think its needed here but in processing thread
                print(str(counter)+"\t"+str(device_name)+"\t"+str(decay.total_seconds()))
                #self.logfile2.write(str(cnt)+"\t"+str(dev)+"\t"+str(thing)+"\t"+str(decay.total_seconds())+"\n")


    def setup_before_detection_gpu(self):
        self.input_size = [int(self.model.net_info['height']), int(self.model.net_info['width'])]
        self.colors = pkl.load(open("pallete", "rb"))
        self.classes = self.load_classes ("data/coco.names")
        self.colors = [self.colors[1]]


    def detection_gpu_return(self, model, frame, cnt):
#        print('Detecting...')
        personcnt =0
        start_time = time.time()
        frame_tensor = cv_image2tensor(frame, self.input_size).unsqueeze(0)
        frame_tensor = Variable(frame_tensor)

        if self.cuda:
            frame_tensor = frame_tensor.cuda()

        detections = self.model(frame_tensor, self.cuda).cpu()
        detections = process_result(detections, self.confidence, self.nms_thresh)
        
        if len(detections) != 0:
            detections = transform_result(detections, [frame], self.input_size)
#            for detection in detections:
            for idx, detection in enumerate(detections):
                if(self.classes[int(detection[-1])]=="person"):
                    personcnt +=1
        print(personcnt)
        return personcnt

    def detection_gpu(self, model, frame, cnt):
#        print('Detecting...')
        start_time = time.time()
        frame_tensor = cv_image2tensor(frame, self.input_size).unsqueeze(0)
        frame_tensor = Variable(frame_tensor)

        if self.cuda:
            frame_tensor = frame_tensor.cuda()

        detections = self.model(frame_tensor, self.cuda).cpu()
        detections = process_result(detections, self.confidence, self.nms_thresh)
        print("number of detected objects: ", len(detections))
        a = [[] for _ in range(len(detections))]
        if len(detections) != 0:
            detections = transform_result(detections, [frame], self.input_size)
#            for detection in detections:
            for idx, detection in enumerate(detections):
                a[idx].append(float(detection[6])) # prediction score
                a[idx].append(self.classes[int(detection[-1])]) # prediction class
                a[idx].append(int(detection[1])) # x1
                a[idx].append(int(detection[2])) # y1
                a[idx].append(int(detection[3])) # x2 
                a[idx].append(int(detection[4])) # y2
#                print(a)
                self.draw_bbox([frame], detection, self.colors, self.classes)
        # save frames if you need to.
#        cv2.imwrite('frame'+cnt+'.jpg', frame)
        end_time = time.time()
        print("[INFO] detection done. It took "+str(end_time-start_time)+" seconds")
        # self.logfile3.write(str(cnt)+"\t"+str(len(detections))+"\t"+str(a)+"\t"+str(end_time-start_time)+"\n")
        self.logfile3.write(str(cnt)+"\t"+str(len(detections))+"\t"+str(a)+"\t"+str(end_time-start_time)+"\n")

    def detection(self, frame):
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300, 300), 127.5)
        print("[INFO] computing object detections...")
        self.net.setInput(blob)
        detections = self.net.forward()
        print("number of objects: ",(detections.shape[2]))
        for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
                confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
                if confidence > self.confidence:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                        label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
                        print("[INFO] {}".format(label))
                        cv2.rectangle(frame, (startX, startY), (endX, endY),self.colors[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[idx], 2)
#        cv2.imshow("Output", frame)
#        cv2.waitkey(0)
        print("[INFO] detection done")

    def load_classes(self, namesfile):
        fp = open(namesfile, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def process_image_metadata(self, msg_dict):
        print(' - image metadata')
        print(' - %s' % msg_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IoT controller of Chameleon.")
    parser.add_argument('-ln1', '--logfilename1', type=str, default='logfiledev1.txt', help="logfile name for dev1")
    parser.add_argument('-ln2', '--logfilename2', type=str, default='logfiledev2.txt', help="logfile name for dev2.")
    parser.add_argument('-ln3', '--logfilename3', type=str, default='logfilecontext.txt', help="logfile name for context related things.")
    parser.add_argument('-ln4', '--logfilename4', type=str, default='logfilegpu.txt', help="logfile name for gpu.")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.6)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.5)
    parser.add_argument('-dis', '--display', type=str, default = 'off', help = "enable display")
    parser.add_argument('-tr', '--transmission', type=str, default = 'dr', help = "e1-1, e1-2, e2, p")
    parser.add_argument('-ts', '--trackingscheme', type=str, default = 'dr', help = "dead reckoning, boundary check")
    parser.add_argument('-fs', '--frameskips', type=int, default = 10, help = "skip frame count")
    ARGS = parser.parse_args()
    # Read 'master.ini'
    config = configparser.ConfigParser()
    config.read('../resource/config/master.ini')
    listen_port = config['controller']['port']
    controller_name = config['controller']['name']
    label_path = config['detection_label']['yolo']
    prototxtpath =  config['mSSD']['prototxt']
    modelpath = config['mSSD']['model']
    ctrl = Controller(controller_name, listen_port)
    print("[INFO] Loading model...")
    print("[INFO] Please wait until setup is done...")
    ctrl.net = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)

    ctrl.confidence = ARGS.confidence
    ctrl.nms_thresh = ARGS.nms_thresh
    ctrl.display = ARGS.display
    ctrl.model = Darknet("cfg/yolov3.cfg")
    ctrl.model.load_weights('yolov3.weights')
    ctrl.cuda = torch.cuda.is_available()
    if torch.cuda.is_available():
        ctrl.model.cuda()
    ctrl.model.eval()
    ctrl.setup_before_detection_gpu()

    ctrl.logfile1 = open(ARGS.logfilename1, 'w')
    ctrl.logfile2 = open(ARGS.logfilename2, 'w')
    ctrl.logfile3 = open(ARGS.logfilename3, 'w')
    ctrl.logfile4 = open(ARGS.logfilename4, 'w')
    ctrl.label_path = label_path
    ctrl.tr = ARGS.transmission
    ctrl.ts = ARGS.trackingscheme
    ctrl.frame_skips = ARGS.frameskips
    print("[INFO] Finished setup!")

    if ARGS.transmission == 'e1-1':
        print('[Controller] running as an existing work 1-1. receiving all frames and strart tracking')
        time.sleep(1)
        ctrl.e1_1_proc_dequeue()

    elif ARGS.transmission == 'e1-2':
        print('[Controller] running as an existing work 1-2. (upon request)')
        time.sleep(1)
        ctrl.e1_2_proc_dequeue()
        
    elif ARGS.transmission == 'e2': # need imp
        print('[Controller] running as an existing work 2. (image metadata)') 
        time.sleep(1)
              
    elif ARGS.transmission == 'p':
        print('[Controller]  tracking objects with dequeuing technique. 1) boundary checking 2) dead reckoning')
        time.sleep(1)
        ctrl.p_proc_dequeue()
    else:
        print('[Controller]  Error: invalid option for the scheme.')

