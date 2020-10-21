import zmq
import asyncio
from zmq.asyncio import Context
import base64
import threading
from  multiprocessing import Process
import cv2
import time
import queue
from math import floor
from statistics import mean
#from streamQueue import streamQueue
import sys
import argparse
from imutils.video import VideoStream
import socket


name2ip_table = {"fogos_server":"143.248.57.73"}

parser = argparse.ArgumentParser(description='parser')

parser.add_argument('--destination',required=True)
parser.add_argument('--start_time',required=False, default='0')
parser.add_argument('--sector',required=False, default='0')


args = parser.parse_args()

ip2name_table = {"143.248.1.1":"ctrl_1", "143.248.1.2":"robot","192.168.1.218":"cam1_1","143.248.57.73":"fogos_server", "143.248.53.69":"cam1_1", "127.0.1.1":"cam1_1"}


local_ip = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)),
s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET,
socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

#local_ip = socket.gethostbyname(socket.getfqdn())
my_name = ip2name_table[local_ip]
camnum = int(my_name[-1])

dest = name2ip_table[args.destination]

time.sleep(int(args.start_time))

#vp = "/home/cdslnab/samplevideo/carla/carla_6cam_{}.avi".format(camnum)
#vp = "/home/spencer1/samplevideo/for_demo/camnet/camnet.mp4"
#vp = "/home/spencer1/samplevideo/for_demo/virat/virat.mp4"
#vp = "/home/cdslnab/samplevideo/carla/carla.avi"

def receivereplies(): #* save it to a local queue. 
    prev_time=time.time()
    firstarrival = True
    while(True):
        #* receive reid objects 
        message = socket_reid_rep.recv_pyobj(flags=0)
        #squeue.enqueue(message['frame'])
        
        if firstarrival == True:

            firstarrival=False

        wc = message['wc']
        if wc == "EOF":
            print("[INFO] finishing client")
            break
    print("releasing video")
    #

portnum = 5555+int(camnum)

context = zmq.Context()

print("Connecting to hello world server…")
socket_reid_req = context.socket(zmq.PUSH)
socket_reid_req.connect("tcp://"+dest+":"+str(portnum))
print("tcp://"+dest+":"+str(portnum))

socket_reid_rep = context.socket(zmq.PULL)
socket_reid_rep.connect("tcp://"+dest+":5555")

#open video files
#video=cv2.VideoCapture(vp)
#input_fps = video.get(cv2.CAP_PROP_FPS)
video = VideoStream(src=0).start()
input_fps = 15

count=0
framecnt=0
# * Create videowriter for output

t2 = threading.Thread(target=receivereplies, args=())
# t2.daemon=True
t2.start()

#squeue = streamQueue()

#  Do 10 requests, waiting each time for a response

while True:
    try:
        start_time = time.time()
        # read video frame
        image=video.read()
        if image is None:
            print("Wrong")
        success = True
        # * shoots every 1/input_fps seconds.
        time.sleep(1/input_fps)

        send_time = time.time()
        if success==False:
            socket_reid_req.send_pyobj(dict(camnum=camnum,frame=image, ts=time.time(), count=count, send_time=send_time, input_fps=input_fps, wc="EOF"))
            break
        else:
            socket_reid_req.send_pyobj(dict(camnum=camnum,frame=image, ts=time.time(), count=count, send_time=send_time, input_fps=input_fps, wc="continue"))
        print("[INFO] sent {} at FPS {}".format(count, input_fps))

        count+=1
        #framecnt+=1
    except:
        print("error")
        video.stream.stream.release()
        socket_reid_req.close()
        break
socket_reid_req.close()

