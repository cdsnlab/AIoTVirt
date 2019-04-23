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
        self.msg_bus.register_callback('img', self.handle_message)
        self.msg_bus.register_callback('img_metadata', self.handle_message)
        self.model = None
        signal.signal(signal.SIGINT, self.signal_handler)
        self.logfile = None
        self.logfile2 = None
        self.logfile3 = None
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
        self.imgq = queue.Queue() # q for image.
        self.timerq = queue.Queue() # q for image wait time
        self.framecntq = queue.Queue() # q for frame cnt
        self.image_dequeue_proc()
        self.gettimegap()

    def gettimegap(self):
        starttime = datetime.now()
        ntp_response = ntplib.NTPClient().request('2.kr.pool.ntp.org', version=3)
        returntime = datetime.now()
        self.timegap = datetime.fromtimestamp(ntp_response.tx_time) - starttime - (returntime - starttime) / 2

    def cpuusage(self):
        self.cpu = psutil.cpu_percent()
        return self.cpu

    def ramusage(self):
        self.ram = psutil.virtual_memory()
        return self.ram

    def signal_handler(self, sig, frame):
        self.logfile.close()
        self.logfile2.close()
        self.logfile3.close()
        print('closing logfile')
        torch.cuda.empty_cache()
        print('clearing cuda cache')
        sys.exit(0)

    def draw_bbox(self, imgs, bbox, colors, classes):
        img = imgs[int(bbox[0])]
        label = classes[int(bbox[-1])]
        #print("label: ", label)
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
        print('[Controller] handle_message: %s' % msg_dict['type'])
        if msg_dict['type'] == 'join':
            self.handle_join(msg_dict)
        elif msg_dict['type'] == 'img': # image raw data
            self.process_raw_image(msg_dict)
        elif msg_dict['type'] == 'img_tracking': # image raw data
            self.process_raw_tracking(msg_dict)
        elif msg_dict['type'] == 'img_metadata': # image raw data
            self.process_image_metadata(msg_dict)
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

    def process_raw_tracking(self, msg_dict):
        print(' - tracking image')
        decstr = base64.b64decode(msg_dict['img_string'])
        imgarray = np.fromstring(decstr, dtype=np.uint8)
        decimg = cv2.imdecode(imgarray, cv2.IMREAD_COLOR)
        localNow = datetime.utcnow()+self.timegap

        curTime = datetime.utcnow().strftime('%H:%M:%S.%f') # string forma
        curdatetime = datetime.strptime(curTime, '%H:%M:%S.%f')
        sentdatetime = datetime.strptime(msg_dict['time'], '%H:%M:%S.%f')

        self.imgq.put(decimg) # keep chugging
        self.timerq.put(curdatetime)
        self.framecntq.put(str(msg_dict['framecnt']))
        self.dev_nameq.put(str(msg_dict['device_name']))

    def process_raw_image(self, msg_dict):
        print(' - raw image')
        decstr = base64.b64decode(msg_dict['img_string'])
        imgarray = np.fromstring(decstr, dtype=np.uint8)
        decimg = cv2.imdecode(imgarray, cv2.IMREAD_COLOR)
#        cv2.imwrite(msg_dict['time']+'.jpg', decimg)
#        print(' - saved img.')
        localNow = datetime.utcnow()+self.timegap
        curTime = datetime.utcnow().strftime('%H:%M:%S.%f') # string forma
        curdatetime = datetime.strptime(curTime, '%H:%M:%S.%f')
        sentdatetime = datetime.strptime(msg_dict['time'], '%H:%M:%S.%f')
        self.logfile.write(str(msg_dict['framecnt'])+"\t"+str((curdatetime - sentdatetime).total_seconds())+"\t"+str(sys.getsizeof(decimg))+'\t'+str(self.cpuusage())+'\n')

        self.imgq.put(decimg) # keep chugging
        self.timerq.put(curdatetime)
        self.framecntq.put(str(msg_dict['framecnt']))
        self.framecnt = str(msg_dict['framecnt'])
    

    @threaded
    def image_dequeue_proc(self):
        prev_time= 0
        endcnt =0 # if idle for 2 minutes, save and quit.
        while (True):
            if (self.imgq.empty()):
                if(endcnt >= 120):
                    self.logfile.close()
                    self.logfile2.close()
                    print("byebye")
                    sys.exit(0)
                else:
                    time.sleep(1)
                    endcnt += 1

                continue
            else:
                cnt = self.framecntq.get()
                dev = self.dev_nameq.get()
                if self.cuda:
                    
                    #self.detection_gpu(self.model, self.imgq.get(), cnt)
                    thing = self.detection_gpu_return(self.model, self.imgq.get(), cnt)
                else: # no GPU enabled
                    self.detection(self.imgq.get())
                curT = datetime.utcnow().strftime('%H:%M:%S.%f') # string format
                decay = datetime.strptime(curT, '%H:%M:%S.%f') - self.timerq.get()
#                print(type(decay))
#                print(decay.total_seconds())
                curr_time = time.time()
                sec = curr_time - prev_time
                prev_time = curr_time
                fps = 1/ (sec)
#                print("fps: ", fps)
                self.logfile2.write(str(cnt)+"\t"+str(dev)+"\t"+str(thing)+str(decay.total_seconds())+"\n")

    def setup_before_detection_gpu(self):
        self.input_size = [int(self.model.net_info['height']), int(self.model.net_info['width'])]
        self.colors = pkl.load(open("pallete", "rb"))
        self.classes = self.load_classes ("data/coco.names")
        self.colors = [self.colors[1]]


    def detection_gpu_return(self, model, frame, cnt):
#        print('Detecting...')
        start_time = time.time()
        frame_tensor = cv_image2tensor(frame, self.input_size).unsqueeze(0)
        frame_tensor = Variable(frame_tensor)

        if self.cuda:
            frame_tensor = frame_tensor.cuda()

        detections = self.model(frame_tensor, self.cuda).cpu()
        detections = process_result(detections, self.confidence, self.nms_thresh)
        print("number of detected objects: ", len(detections))
        return len(detections)

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
    parser.add_argument('-ln1', '--logfilename1', type=str, default='logfilecpu.txt', help="logfile name for cpu usage")
    parser.add_argument('-ln2', '--logfilename2', type=str, default='logfileframe.txt', help="logfile name for frame related things.")
    parser.add_argument('-ln3', '--logfilename3', type=str, default='logfilecontext.txt', help="logfile name for context related things.")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.7)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.5)
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
    ctrl.model = Darknet("cfg/yolov3.cfg")
    ctrl.model.load_weights('yolov3.weights')
    ctrl.cuda = torch.cuda.is_available()
    if torch.cuda.is_available():
        ctrl.model.cuda()
    ctrl.model.eval()
    ctrl.setup_before_detection_gpu()

    ctrl.logfile = open(ARGS.logfilename1, 'w')
    ctrl.logfile2 = open(ARGS.logfilename2, 'w')
    ctrl.logfile3 = open(ARGS.logfilename3, 'w')
    ctrl.label_path = label_path
    print("[INFO] Finished setup!")

