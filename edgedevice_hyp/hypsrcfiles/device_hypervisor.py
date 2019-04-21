import argparse
import configparser
import sys
import time
from datetime import datetime,date
sys.path.insert(0, '../../messaging')
from message_bus import MessageBus
from utils import visualize_output
from utils import deserialize_output
import mvnc.mvncapi as mvnc
import dlib
import json
import redis
import cv2
import numpy as np
import psutil
import ast
import threading
import imutils
from imutils.object_detection import non_max_suppression
import signal
import ntplib
import trackableobject 
import centroidtracker


#
# Reads a graph file into a buffer
#
def load_graph(graph_file, device):
    with open(graph_file, mode='rb') as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph(blob)
    return graph

#
# Decorator for threading methods in a class
#
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class Hypervisor(object):
    def __init__(self, name, port, ip_ext, port_ext, ifname, controller_ip, controller_port, live, tr_method):
        self.device_name = name
        self.device_port = port
        self.device_ip_ext = ip_ext
        self.device_port_ext = port_ext
        self.ifname = ifname
        self.controller_ip = controller_ip
        self.controller_port = controller_port

        self.msg_bus = MessageBus(name, port, 'camera')
        self.msg_bus.register_callback('join', self.handle_message)
        self.msg_bus.register_callback('device_list', self.handle_message)
        self.msg_bus.register_callback('handoff_request', self.handle_message)
        signal.signal(signal.SIGINT, self.signal_handler)

        self.camera = None  # OpenCV camera object
        self.live = live
        self.tr_method = tr_method
        self.labels = None
        self.confidence_threshold = 0.0
        self.redis_db = None
        self.display = 'off'
        self.graph_file = ''
        self.width = 600
        self.height = 400
        self.counter = 0
        self.color_mode = 'bgr'
        self.dimensions = [224, 224]
        self.mean = [127.5, 127.5, 127.5]
        self.scale = 0.00789
        self.starttime = time.time()
        self.logfile = None
        self.encode_param = None
#        self.timegap = datetime.datetime()
        self.gettimegap()
        self.curframe = None # current frame for being cropped
        self.tm_methods = ['cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR', 'cv2.TM_SQDIFF']
        self.tm_op1 = cv2.TM_CCOEFF_NORMED
        self.tm_op2 = cv2.TM_CCOEFF
        self.tm_op3 = cv2.TM_CCORR_NORMED
        self.tm_op4 = cv2.TM_CCORR
        self.tm_op5 = cv2.TM_SQDIFF_NORMED
        self.tm_op6 = cv2.TM_SQDIFF
        self.frame_skips = None # how many frames should be skipped before detection 
        self.ct = None # centroid tracker
        self.trackers = [] 
        self.trobs = {} # tracking objects
        self.boundary = {} # check if its on the boundary of the frame 0: top, 1: right, 2: bottom, 3: left
        self.objstatus = {} # objects current moving direction
        self.framethr = 0 # boundary of the frame to indicate the the its leaving 
        self.sumframebytes = 0
        self.findobj = False # if handoff request is recieved it needs ot find the object in the frame
        self.template =None # cropped image of the object


    def gettimegap(self):
        starttime = datetime.now()
        ntp_response = ntplib.NTPClient().request('2.kr.pool.ntp.org', version=3)
        returntime = datetime.now()
        self.timegap = datetime.fromtimestamp(ntp_response.tx_time) - starttime - (returntime - starttime)/2

    def cpuusage(self):
        return psutil.cpu_percent()

    def ramusage(self):
        return psutil.virtual_memory()

    def signal_handler(self, sig, frame):
        self.logfile.close()
        print('closing logfile, exiting')
        sys.exit(0)

    def load_labels(self, labels_file):
        self.labels = [ line.rstrip('\n') for line in
              open(labels_file) if line != 'classes\n']

    def handle_message(self, msg_dict):
        print('[Hypervisor] handle_message: %s' % msg_dict['type'])
        if msg_dict['type'] == 'join':
            self.handle_join(msg_dict)
        elif msg_dict['type'] == 'device_list':
            # print(msg_dict)
            device_list = json.loads(msg_dict['devices'])
            for item in device_list:
                if item['device_name'] == self.device_name:
                    continue
                else:
                    # adding a node info that i am not aware of...
                    print(' - adding a new node_info: ', item['device_name'])
                    self.msg_bus.node_table.add_entry(item['device_name'], item['ip'], item['port'], item['location'], item['capability'])

        elif msg_dict['type'] == 'handoff_request':
            self.process_cropped_image_tracking(msg_dict)
            #self.findobj==True
            # add the image to trackable list... but we don't know the coordinates...!

        else:
            # Silently ignore invalid message types.
            pass


    def process_cropped_image_tracking(self, msg_dict):
        print(' - received handoff request')

        decstr = base64.b64decode(msg_dict['img_string'])
        imgarray = np.fromstring(decstr, dtype=np.uint8)
        tracking_template = cv2.imdecode(imgarray, cv2.IMREAD_COLOR)
        for meth in self.tm_methods:
            res = cv2.matchTemplate(self.curframe, tracking_template, meth)
            print(res)
            min_val,max_val,min_loc, max_loc = cv2.minMaxLoc(res) # max_val is the matching threshold
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0]+w,top_left[1]+h)
            cv2.rectangle(self.curframe, top_left, bottom_right,255,5)
            cv2.imshow("lets track", frame)


        localNow = datetime.utcnow()+self.timegap
        curTime = datetime.utcnow().strftime('%H:%M:%S.%f') # string forma
        curdatetime = datetime.strptime(curTime, '%H:%M:%S.%f')
        sentdatetime = datetime.strptime(msg_dict['time'], '%H:%M:%S.%f')
        '''
        self.imgq.put(decimg) # keep chugging
        self.timerq.put(curdatetime)
        self.framecntq.put(str(msg_dict['framecnt']))
        self.framecnt = str(msg_dict['framecnt'])
        '''

    def join(self):
        # Create a join message based on NIC information.

        print("connecting to edge server")
        join_msg = dict(type='join', device_name=self.device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
        self.msg_bus.send_message_json(self.controller_ip, self.controller_port, join_msg)
        
        # We might not need to join other cameras, just sent handoff-message!

        
        if self.device_name == "camera02": #if this devices is center device
            print("connecting to left cam")
            join_msg = dict(type='join', device_name=device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
            self.msg_bus.send_message_json(self.left_device_ip_int, self.left_device_port, join_msg)
            print("connecting to right cam")
            join_msg = dict(type='join', device_name=device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
            self.msg_bus.send_message_json(self.right_device_ip_int, self.right_device_port, join_msg)

        elif self.device_name == "camera01": # if this devices is the left device
            print("connecting to center cam")
            join_msg = dict(type='join', device_name=device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
            self.msg_bus.send_message_json(self.center_device_ip_int, self.center_device_port, join_msg)
            '''
            print("connecting to right cam")
            join_msg = dict(type='join', device_name=device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
            self.msg_bus.send_message_json(self.right_device_ip_int, self.right_device_port, join_msg)


        elif self.device_name == "camera03": # if this device is the right device
            print("connecting to left cam")
            join_msg = dict(type='join', device_name=device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
            self.msg_bus.send_message_json(self.left_device_ip_int, self.left_device_port, join_msg)
            print("connecting to right cam")
            join_msg = dict(type='join', device_name=device_name, ip=self.device_ip_ext, port=self.device_port_ext,
                        location='N1_823_1', capability='no')
            self.msg_bus.send_message_json(self.center_device_ip_int, self.center_device_port, join_msg)
            '''
    def handle_join(self, msg_dict):
        node_table = self.msg_bus.node_table
        node_table.add_entry(msg_dict['device_name'], msg_dict['ip'], int(msg_dict['port']), msg_dict['location'], msg_dict['capability'])
        print('@@Table: ', self.msg_bus.node_table.table)

        # Send node_table
        device_list_json = {"type": "device_list", "devices": self.msg_bus.node_table.get_list_str()}
        for node_name in node_table.table.keys():
            node_info = node_table.table[node_name]
            self.msg_bus.send_message_json(node_info.ip, int(node_info.port), device_list_json)

    def connect_redis_db(self, redis_port):
        self.redis_db = redis.Redis(host='localhost', port=redis_port, db=0)

    def open_ncs_device(self):
        # Look for enumerated NCS device(s); quit program if none found.
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print("No devices found")
            quit()
        # Get a handle to the first enumerated device and open it
        device = mvnc.Device(devices[0])
        device.OpenDevice()
        return device

    def getfps(self, oldtime):
        curr_time = time.time()
        sec = curr_time - oldtime
        fps = 1 / sec
        return curr_time, fps
        

    def close_ncs_device(self, device, graph):
        graph.DeallocateGraph()
        device.CloseDevice()
        self.camera.release()
        cv2.destroyAllWindows()

    def pre_process_image(self, frame):
        # Resize image [Image size is defined by chosen network, during training]
        img = cv2.resize(frame, tuple(self.dimensions))

        # Convert RGB to BGR [OpenCV reads image in BGR, some networks may need RGB]
        if (self.color_mode == "rgb"):
            img = img[:, :, ::-1]

        # Mean subtraction & scaling [A common technique used to center the data]
        img = img.astype(np.float16)
        img = (img - np.float16(self.mean)) * self.scale

        return img

    #cascade here..
    def infer_image_haar(frame, fps):
        a = []
        curTime = time.time()
        body_cascade = cv2.CascadeClassifier('cascades/haarcascade_upperbody.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        body = body_cascade.detectMultiScale(gray, 1.1, 8)
        infTime = time.time()-curTime 

        a = [[] for _ in range(len(pick))]
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()

        for i in len(body):

            for(xA, yA, xB, yB) in pick:
                a[i].append("90")
                a[i].append("15:person")
                (y1, x1) = (yA, xA)
                a[i].append(y1,x1)
                (y2, x2) = (yB, xB)
                a[i].append(y2,x2)

        save = {"elapsedtime": "{0:.2f}".format(elapsedtime), "CPU": str(cpu), "inftime": str("{0:.2f}".format(inftime)), "fps": str("{0:.2f}".format(fps)), "numberofobjects": str(len(pick)),"a": str(a)}
        r.hmset(counter, save)
        del(a)



    # hog codes here
    def infer_image_hog (frame, fps):
        a = []  
        
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        curTime = time.time()
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
        rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        infTime = time.time()-curTime 
       
        a = [[] for _ in range(len(pick))]
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        for i in len(pick):
    
            for(xA, yA, xB, yB) in pick:
                a[i].append("90")
                a[i].append("15:person")
                (y1, x1) = (yA, xA)
                a[i].append(y1,x1)
                (y2, x2) = (yB, xB)
                a[i].append(y2,x2)

        self.counter += 1

        save = {"elapsedtime": "{0:.2f }".format(elapsedtime), "CPU": str(cpu), "inftime": str("{0:.2f}".format(inftime)), "fps": str("{0:.2f}".format(fps)), "numberofobjects": str(len(pick)),"a": str(a)}
        r.hmset(counter, save)
        del(a)

    def infer_image_fps(self, graph,img, frame, fps):
        # Load the image as a half-precision floating point array
        graph.LoadTensor(img, 'user object')

        # Get the results from NCS
        output, userobj = graph.GetResult()

        # Get execution time
        inference_time = graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)

        # Deserialize the output into a python dictionary
        output_dict = deserialize_output.ssd(
            output,
            self.confidence_threshold,
            frame.shape)

        # print( "I found these objects in ( %.2f ms ):" % ( numpy.sum( inference_time ) ) )
        inftime = np.sum(inference_time)
        numobj = (output_dict['num_detections'])

        # create array for detected obj
        a = [[] for _ in range(numobj)]

        # print (numobj)
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()

        for i in range(0, output_dict['num_detections']):
            print("%3.1f%%\t" % output_dict['detection_scores_' + str(i)]
                  + self.labels[int(output_dict['detection_classes_' + str(i)])]
                  + ": Top Left: " + str(output_dict['detection_boxes_' + str(i)][0])
                  + " Bottom Right: " + str(output_dict['detection_boxes_' + str(i)][1]))
            #        print(str(i))
            a[i].append(output_dict['detection_scores_' + str(i)])
            a[i].append(self.labels[int(output_dict['detection_classes_' + str(i)])])
            a[i].append(str(output_dict['detection_boxes_' + str(i)][0]))
            a[i].append(str(output_dict['detection_boxes_' + str(i)][1]))
            # Draw bounding boxes around valid detections
            (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
            (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

            # Prep string to overlay on the image

            display_str = (self.labels[output_dict.get('detection_classes_' + str(i))] + ": " + str(
                output_dict.get('detection_scores_' + str(i))) + "%")

            frame = visualize_output.draw_bounding_box(
                y1, x1, y2, x2,
                frame,
                thickness=4,
                color=(255, 255, 0),
                display_str=display_str)
            cv2.putText(frame, 'FPS:' + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                        cv2.LINE_AA)
        #        cv2.putText(frame, direction, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255),3)
        #    print( '\n' )

        # If a display is available, show the image on which inference was performed
        self.counter += 1
        if self.display == "on":
            cv2.imshow('NCS live inference', frame)

        # need to save to redis.
        elapsedtime = time.time() - self.starttime
        save = {"elapsedtime": "{0:.2f}".format(elapsedtime), "CPU": str(cpu),
                "inftime": str("{0:.2f}".format(inftime)), "fps": str("{0:.2f}".format(fps)),
                "numberofobjects": str(numobj), "a": str(a)}

        self.redis_db.hmset(self.counter, save)
        self.logfile.write(str(self.counter))
        self.logfile.write(str(save)+"\n")
        #print(self.redis_db.hgetall(self.counter))
        # print(save)
        # need plots...! for multiple objects
        del (a)
        return numobj

    #
    # Existing Work 1 (E1): sending only raw images
    #
    def img_ssd_send_raw_image(self):
        framecnt = 0
        prev_time = 0
        # make ncs connection
        device = self.open_ncs_device()
        graph = load_graph(self.graph_file, device)

        # Main loop: Capture live stream & send frames to NCS
        if self.live == str(1):
            self.camera = cv2.VideoCapture(0)
            while (True):
                ret, frame = self.camera.read()
                #### get fps                
                prev_time, fps = self.getfps(prev_time)
                print("estimated live fps {0}".format(fps))
#                img = self.pre_process_image(frame)
                smallerimg = cv2.resize(frame, (self.width, self.height))
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                # log here.
                self.logfile.write(str(framecnt)+"\t"+str(sys.getsizeof(smallerimg))+"\t"+str(cpu)+"\n")
                jsonified_data = MessageBus.create_message_list_numpy(smallerimg, framecnt, self.encode_param, self.device_name,self.timegap)
                self.msg_bus.send_message_str(self.controller_ip, self.controller_port, jsonified_data)
                framecnt += 1


                # Display the frame for 5ms, and close the window so that the next
                # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
                
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

            self.close_ncs_device(device, graph)
        # sy: read video from file
        else:
            self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            cap = cv2.VideoCapture(self.live)

            while cap.isOpened():
#                curTime = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
                curTime=datetime.utcnow().strftime('%H:%M:%S.%f')
                ret, frame = cap.read()  # ndarray
                prev_time, fps = self.getfps(prev_time)
                print("estimated transmission fps {0}".format(fps))
                img = self.pre_process_image(frame)
                #result, encimg = cv2.imencode('.jpg', smallerimg, self.encode_param)
                if (ret!=1):
                    self.logfile.close()
                    sys.exit(0)
            
                smallerimg = cv2.resize(img, (self.width, self.height))
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory()
                # log here.
                self.logfile.write(str(framecnt)+"\t"+str(sys.getsizeof(smallerimg))+"\t"+str(cpu)+"\n")
                jsonified_data = MessageBus.create_message_list_numpy(smallerimg, framecnt, self.encode_param, self.device_name,self.timegap)
                self.msg_bus.send_message_str(self.controller_ip, self.controller_port, jsonified_data)
                framecnt += 1

                if (cv2.waitKey(3) & 0xFF == ord('q')):
                    break
#            cap.release()


#
# existing work e2
#
    @threaded
    def img_ssd_save_send_metadata(self):
        framecnt = 0
        prev_time = 0

        # make ncs connection
        device = self.open_ncs_device()
        graph = load_graph(self.graph_file, device)

        # Main loop: Capture live stream & send frames to NCS
        if self.live == str(1):
            self.camera = cv2.VideoCapture(0)
            while (True):
                ret, frame = self.camera.read()
                #### get fps
                prev_time, fps = self.getfps(prev_time)

                print("estimated live fps {0}".format(fps))
                img = self.pre_process_image(frame)
                # this is spencers code for infering fps.
                self.infer_image_fps(graph, img, frame, fps)

                # Display the frame for 5ms, and close the window so that the next
                # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

            self.close_ncs_device(device, graph)

        else:
            self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            cap = cv2.VideoCapture(self.live)

            while cap.isOpened():
                curr_time_str = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
                ret, frame = cap.read()  # ndarray
                smallerimg = cv2.resize(frame, (self.width, self.height))
                # result, encimg = cv2.imencode('.jpg', smallerimg, encode_param)


                # TODO: Capture contexts.
                #### get fps
                prev_time, fps = self.getfps(prev_time)
                print("estimated video fps {0}".format(fps))
                img = self.pre_process_image(smallerimg)
                self.infer_image_fps(graph, img, smallerimg, fps)

                self.img_ssd_send_metadata(framecnt)
                framecnt += 1
                if (cv2.waitKey(3) & 0xFF == ord('q')):
                    break
#            cap.release()

    def img_ssd_send_metadata(self, framecnt):
#        print('[Hypervisor] Existing work 2: load and send metadata')
        localNow = datetime.utcnow()+self.timegap
        curTime = localNow.strftime('%H:%M:%S.%f') # string format
        # load metadata from Redis
        save=self.redis_db.hgetall(self.counter)
        save.update({'type': 'img_metadata'})
        save.update({'framecnt': framecnt})
        save.update({'time': curTime})
        print(save)
#            contexts = {'a': 'a'}

            
#            curr_time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(' -', curTime)
#            metadata_json = {'type': 'img_metadata', 'device_name': self.device_name, 'context': contexts, 'time': curTime}
        self.msg_bus.send_message_json(self.controller_ip, self.controller_port, save)
        time.sleep(0.001)

    @threaded
    def tracking_objects(self):
        framecnt = 0
        prev_time = 0

        # make ncs connection
        device = self.open_ncs_device()
        graph = load_graph(self.graph_file, device)

        # Main loop: Capture live stream & send frames to NCS
        if self.live == str(1):
            while (True):
                ret, frame = self.camera.read()
                #### get fps
                prev_time, fps = getfps(prev_time)

                print("estimated live fps {0}".format(fps))
                img = self.pre_process_image(frame)
                # this is spencers code for infering fps.
                # self.infer_image_fps(graph, img, frame, fps)
                self.periodic_tracking(graph)

                # Display the frame for 5ms, and close the window so that the next
                # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

            self.close_ncs_device(device, graph)

        else:

            # detects objects every 10 frames, tracks every frames.
            self.periodic_tracking(graph)

    def checkboundary(self, centroid, objectID):
        if(centroid[0]<self.framethr):
            if(centroid[1]<self.framethr):
                self.boundary[objectID] = "TL"
            elif(centroid[1]>self.framethr and centroid[1] < (self.height-self.framethr)):
                self.boundary[objectID] = "CL"
            elif(centroid[1]>(self.width-self.framethr)):
                self.boundary[objectID] = "BL"
        elif (centroid[0]>self.framethr and centroid[0] < (self.width - self.framethr)):
            if(centroid[1] < self.framethr):
                self.boundary[objectID] = "TC"
            elif(centroid[1]>self.framethr and centroid[1] < (self.height-self.framethr)):
                self.boundary[objectID] = "CC"
            elif(centroid[1]>(self.width-self.framethr)):
                self.boundary[objectID] = "BC"
        elif (centroid[0] > (self.width - self.framethr)):
            if(centroid[1] < self.framethr):
                self.boundary[objectID] = "TR"
            elif(centroid[1]>self.framethr and centroid[1] < (self.height-self.framethr)):
                self.boundary[objectID] = "CR"
            elif(centroid[1]>(self.width-self.framethr)):
                self.boundary[objectID] = "BR"
        
    def checkdir(self, dirX, dirY,objectID): # this -2, 2 must also be adjusted depending on the tracked objects
        if dirY < -2:
            if dirX < -2:
                self.objstatus[objectID] = "NW"
            elif dirX < 2 and dirX > -2:
                self.objstatus[objectID] = "N"
            elif dirX > 2:
                self.objstatus[objectID] = "NE"
                            
        elif dirY < 2 and dirY > -2:
            if dirX < -2:
                self.objstatus[objectID] = "W"
            elif dirX < 2 and dirX > -2:
                self.objstatus[objectID] = "-"
            elif dirX > 2:
                self.objstatus[objectID] = "E"

        elif dirY > 2:
            if dirX < -2:
                self.objstatus[objectID] = "SW"
            elif dirX < 2 and dirX > -2:
                self.objstatus[objectID] = "S"
            elif dirX > 2:
                self.objstatus[objectID] = "SE"

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


    def periodic_tracking(self, graph):

        prev_time = 0
        labels = []

        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        cap = cv2.VideoCapture(self.live)

        while cap.isOpened():
            curr_time_str = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
            ret, frame = cap.read()  # ndarray
            if frame is None:
                break
            frame = cv2.resize(frame, (self.width, self.height))
            self.curframe = frame
            img = self.pre_process_image(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prev_time, fps = self.getfps(prev_time)
            print("estimated video fps {0}".format(fps))
            positions = []
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            if self.findobj == True:
                print('looking for objects in frame...')
                res = cv2.matchTemplate(frame, self.template, self.tm_op1)
                print(res)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc (res)
                w, h = self.template.shape[::-1]
                br = (min_loc[0] + w, min_loc[1] +h)
                cv2.rectangle(frame, min_loc[0], br, 255,2)

            if self.counter % self.frame_skips ==0:
                self.trackers = []
                graph.LoadTensor(img, 'user object')

                # Get the results from NCS
                output, userobj = graph.GetResult()
                inference_time = graph.GetGraphOption(mvnc.GraphOption.TIME_TAKEN)

                # Deserialize the output into a python dictionary
                output_dict = deserialize_output.ssd(output, self.confidence_threshold, frame.shape)
                

                for i in range(0, output_dict['num_detections']):
                    if output_dict['detection_scores_' + str(i)] > self.confidence_threshold :
                        if output_dict ['detection_classes_'+str(i)] != 15: # skip if not human
                            continue
                        print("%3.1f%%\t" % output_dict['detection_scores_' + str(i)] + self.labels[int(output_dict['detection_classes_' + str(i)])] + ": Top Left: " + str(output_dict['detection_boxes_' + str(i)][0]) + " Bottom Right: " + str(output_dict['detection_boxes_' + str(i)][1]))
                  
                        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
                        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]
                        #print("centroid X Y", (x1+x2) / 2.0, (y1+y2)/2.0)
                        # add to tracking objects
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(x1, y1, x2, y2)
                        tracker.start_track(rgb,rect)
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
            # prints all existing indexes.
            #self.ct.getall()

            for (objectID, centroid) in objects.items():
                to = self.trobs.get(objectID, None)
                #print(self.ct.get_object_rect_by_id(objectID))

                if to == None:
                    to = trackableobject.TrackableObject(objectID, centroid)
                else:

                    if (sendablecnt ==0):
                        self.sumframebytes+=sys.getsizeof(frame)
                    cv2.circle(frame, (centroid[0],centroid[1]),4,(255,255,255),-1)
                    y = [c[1] for c in to.centroids]
                    x = [c[0] for c in to.centroids]
                    dirY = centroid[1] - np.mean(y)
                    dirX = centroid[0] - np.mean(x)
                    to.centroids.append(centroid)
                    
                    if not to.counted:
                        #print("centroid (x,y): ", centroid[0], centroid[1])
                        self.checkboundary(centroid, objectID)
                        #print("boundary: ", self.boundary)

                        self.checkdir(dirX, dirY, objectID)
                        #print("objstatus: ", self.objstatus)

                        where = self.checkhandoff(objectID)
                        # send hand off msg here
                        if(where == "RIGHT"):
                            print("we need to send msg to right")
                            #print("just throw in the cropped img template")
                            #print("upon receiving the cropped img, do the template matching & add to tracking dlib queue")
                            p = self.ct.get_object_rect_by_id(objectID) # x1, y1, x2, y2
                            #t = self.ct.objects[objectID]  #centroid x, y
                            #print("type p: ", type(p)) # rect
                            #print("t: ", t) # centroid
                            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255,0,0),2)
                            #croppedimg = frame[y1:y2, x1: x2]
                            jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, self.encode_param, self.device_name, self.timegap)
                            self.msg_bus.send_message_str(self.center_device_ip_ext, self.center_device_port, jsonified_data)
                            
                        elif (where == "LEFT"):
                            print("we need to send msg to left")
                            #jsonified_data = MessageBus.create_message_list_numpy_handoff(croppedimg, encode_param, self.device_name, self.timegap)
                            #self.msg_bus.send_message_str(self.right_device_ip_ext self.right_device_port, jsonified_data)
                        elif (where == "TOP"):
                            print("we need to send msg to top")
                        elif (where == "BOTTOM"):
                            print("we need to send msg to bottom")
                        else:
                            print("")
                            #print("Nothing is happening")
                    sendablecnt += 1

                self.trobs[objectID] = to
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2) # ID of the object 
            sendablecnt = 0
            self.counter += 1
#            cv2.imwrite('frame'+str(self.counter)+'.jpg', frame)
            if self.display == "on":
                cv2.imshow('NCS live inference', frame)
            if(cv2.waitKey(3) & 0xFF == ord('q')):
                break
            #save = {"CPU": str(cpu), "fps": str("{0:.2f}".format(fps)), "cur_tracking_count": str(len(self.trackers)), "totalbytes": str(self.sumframebytes)}
            self.logfile.write(str(self.counter)+"\t") 
            self.logfile.write(str(cpu)+"\t"+str("{0:.2f}".format(fps))+"\t"+str(len(self.trackers))+"\t"+str(self.sumframebytes)) 
            self.logfile.write("\n")
      
            
        cap.release()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="IoT Camera (device) of Chameleon.")
    parser.add_argument('-i', '--iface', type=str,
                        default='eth0',
                        help="A network interface name for edge network connection (eth0, wlan0, ...)")
    parser.add_argument('-g', '--graph', type=str,
                        default='../SSD_MobileNet/graph',
                        help="Absolute path to the neural network graph file.")
    parser.add_argument('-l', '--labels', type=str,
                        default='../SSD_MobileNet/labels.txt',
                        help="Absolute path to labels file.")
    parser.add_argument('-tr', '--transmission', type=str,
                        default="e1",
                        help="frame transmission options (proposed=p, existing_1=e1, ...)")
    parser.add_argument('-w', '--width', type=int,
                        default="600",
                        help="width of the capturing videos.")
    parser.add_argument('-hi', '--height', type=int,
                        default="400",
                        help="height of the capturing videos.")
    parser.add_argument('-vf', '--videofile', type=str,
                        default="1",
                        help="load from video file.")
    parser.add_argument('-dis', '--display', type=str,
                        default="off",
                        help="load from video file.")
    parser.add_argument('-D', '--dim', type=int,
                        nargs='+',
                        default=[224, 224],
                        help="Image dimensions. ex. -D 224 224")
    parser.add_argument('-c', '--colormode', type=str,
                        default="bgr",
                        help="RGB vs BGR color sequence. This is network dependent.")
    parser.add_argument('-ln', '--logname', type=str,
                        default='logfile.txt',
                        help="your log filename name.")
    parser.add_argument('-sk', '--skipcount', type=int,
                        default=10,
                        help="number of skipping frames for object detection.")
    parser.add_argument('-dt', '--disappear_thr', type=int,
                        default=20,
                        help="number of frames until frame is regarded as disappeared from tracking list.")
    parser.add_argument('-bt', '--boundary_thr', type=int,
                        default=80,
                        help="boundary to regard as the object is leaving the frame.")

    ARGS = parser.parse_args()

    # Read 'camera.ini'
    config = configparser.ConfigParser()
    config.read('../../resource/config/camera_823_main.ini')

    # Hypervisor initialization and connection
    controller_ip = config['message_bus']['controller_ip']
    controller_port = config['message_bus']['controller_port']
    device_name = config['message_bus']['left_device_name']
    device_port = config['message_bus']['left_device_port']
    device_ip_ext = config['message_bus']['left_device_ip_ext']
    device_port_ext = config['message_bus']['left_device_port_ext']
    hyp = Hypervisor(device_name, device_port, device_ip_ext, device_port_ext, ARGS.iface, controller_ip, controller_port, ARGS.videofile, ARGS.transmission)

    #connection to other cameras
    hyp.center_device_name = config['message_bus']['center_device_name']
    hyp.center_device_port = config['message_bus']['center_device_port']
    hyp.center_device_ip_int = config['message_bus']['center_device_ip_int']
    hyp.center_device_ip_ext = config['message_bus']['center_device_ip_ext']

    
    hyp.right_device_name = config['message_bus']['right_device_name']
    hyp.right_device_port = config['message_bus']['right_device_port']
    hyp.right_device_ip_int = config['message_bus']['right_device_ip_int']
    hyp.right_device_ip_ext = config['message_bus']['right_device_ip_ext']
    

    hyp.join()
    # Camera-related settings

    hyp.display = ARGS.display
    hyp.graph_file = ARGS.graph
    hyp.width = ARGS.width
    hyp.height = ARGS.height
    hyp.confidence_threshold = float(config['Parameter']['confidence_threshold'])
    hyp.color_mode = config['Parameter']['color_mode']
    hyp.dimensions = ast.literal_eval(config['Parameter']['dimensions'])
    hyp.mean = ast.literal_eval(config['Parameter']['mean'])
    hyp.scale = float(config['Parameter']['scale'])
    hyp.connect_redis_db(6379)
    hyp.load_labels(ARGS.labels)
    hyp.logfile = open(ARGS.logname, 'w')
    hyp.frame_skips = ARGS.skipcount
    hyp.framethr = ARGS.boundary_thr
    hyp.ct = centroidtracker.CentroidTracker(maxDisappeared=ARGS.disappear_thr, maxDistance =50)

    # Operations based on scheme options
    if ARGS.transmission == 'p':
        print('[Hypervisor] running our proposed scheme.')
        #not properly imp
    elif ARGS.transmission == 'e1':
        print('[Hypervisor] running as an existing work 1. (raw image stream)')
        # Run video analytics with SSD
        hyp.img_ssd_send_raw_image()
        hyp.logfile.close()
        
    elif ARGS.transmission == 'e2': # what is meta data? 
        print('[Hypervisor] running as an existing work 2. (image metadata)') 
        hyp.img_ssd_save_send_metadata()
#        hyp.logfile.close()
        
    elif ARGS.transmission == 'e3':
        print('[Hypervisor] running as an existing work 3.')
        hyp.logfile.close()
    elif ARGS.transmission == 'tracking':
        print('[Hypervisor] tracking objects.')
        hyp.tracking_objects()
        #hyp.logfile.close()
    else:
        print('[Hypervisor] Error: invalid option for the scheme.')
