import argparse
import configparser
import sys
sys.path.insert(0, '../messaging')
from message_bus import MessageBus
from datetime import datetime
import base64
import cv2
import numpy as np
import signal
import imutils
import psutil
import queue
import threading
import time

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
        signal.signal(signal.SIGINT, self.signal_handler)
        self.logfile = None
        self.logfile2 = None
        self.dalgorithm = "yolo"
        self.starttime = 0.0
        self.endtime = 0.0
        self.cpu = None
        self.ram = None
        self.label_path = None
        self.CLASSES = None
        self.COLORS = None
        self.confidence = 0.4
        self.net = None # load caffe model
        self.framecnt = 0
        self.imgq = queue.Queue() # q for image.
        self.timerq = queue.Queue() # q for image wait time
        self.framecntq = queue.Queue() # q for frame cnt
        self.image_dequeue_proc() 

    def cpuusage(self):
        self.cpu = psutil.cpu_percent()
        return self.cpu

    def ramusage(self):
        self.ram = psutil.virtual_memory()
        return self.ram


    def signal_handler(self, sig, frame):
        self.logfile.close()
        self.logfile2.close()
        print('closing logfile')
        sys.exit(0)

    def handle_message(self, msg_dict):
        print('[Controller] handle_message: %s' % msg_dict['type'])
        if msg_dict['type'] == 'join':
            self.handle_join(msg_dict)
        elif msg_dict['type'] == 'img': # image raw data
            self.process_raw_image(msg_dict)
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

    def process_raw_image(self, msg_dict):
        print(' - raw image')
        decstr = base64.b64decode(msg_dict['img_string'])
        imgarray = np.fromstring(decstr, dtype=np.uint8)
        decimg = cv2.imdecode(imgarray, cv2.IMREAD_COLOR)
#        cv2.imwrite(msg_dict['time']+'.jpg', decimg)
#        print(' - saved img.')
        curTime = datetime.utcnow().strftime('%H:%M:%S.%f') # string forma
        curdatetime = datetime.strptime(curTime, '%H:%M:%S.%f')
        sentdatetime = datetime.strptime(msg_dict['time'], '%H:%M:%S.%f')
        self.logfile.write(str(msg_dict['framecnt'])+"\t"+str((sentdatetime - curdatetime).total_seconds())+"\t"+str(sys.getsizeof(decimg))+'\t'+str(self.cpuusage())+'\n')

        self.imgq.put(decimg) # keep chugging
        self.timerq.put(curdatetime)
        self.framecntq.put(str(msg_dict['framecnt']))
    

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

                self.detection(self.imgq.get())
                curT = datetime.utcnow().strftime('%H:%M:%S.%f') # string format
                decay = datetime.strptime(curT, '%H:%M:%S.%f') - self.timerq.get()
#                print(type(decay))
#                print(decay.total_seconds())
                cnt = self.framecntq.get()
                curr_time = time.time()
                sec = curr_time - prev_time
                prev_time = curr_time
                fps = 1/ (sec)
#                print("fps: ", fps)
                self.logfile2.write(str(cnt)+"\t"+str(fps)+"\t"+str(decay.total_seconds())+"\n")


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
                        cv2.rectangle(frame, (startX, startY), (endX, endY),self.COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
#        cv2.imshow("Output", frame)
#        cv2.waitkey(0)
        print("[INFO] detection done")

    def setcc(self):
        print("[INFO] loading classes and colors...")
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def loadmodel(self):
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def load_yolo_labels(self, labels_file):
        self.labels = [line.rstrip('\n') for line in open(labels_file) if line != 'classes\n']

    def process_image_metadata(self, msg_dict):
        print(' - image metadata')
        print(' - %s' % msg_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IoT controller of Chameleon.")
    parser.add_argument('-ln1', '--logfilename1', type=str, default='logfilecpu.txt', help="logfile name for cpu usage")
    parser.add_argument('-ln2', '--logfilename2', type=str, default='logfileframe.txt', help="logfile name for frame related things.")
    ARGS = parser.parse_args()
    # Read 'master.ini'
    config = configparser.ConfigParser()
    config.read('../resource/config/master.ini')
    listen_port = config['controller']['port']
    controller_name = config['controller']['name']
#    logfile_name = config['logs']['logfilename'] # for cpu usage
#    logfile_name2 = config['logs']['logfilename2'] # for frame related info
    label_path = config['detection_label']['yolo']
    prototxtpath =  config['mSSD']['prototxt']
    modelpath = config['mSSD']['model']
    ctrl = Controller(controller_name, listen_port)
    print("[INFO] Loading model...")
    ctrl.net = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)
    ctrl.setcc()
    ctrl.logfile = open(ARGS.logfilename1, 'w')
    ctrl.logfile2 = open(ARGS.logfilename2, 'w')
    ctrl.label_path = label_path


