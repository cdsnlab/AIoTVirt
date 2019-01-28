import sys, zmq
import time
import threading
import base64
import cv2
import io
import json
from PIL import Image
import pickle
import numpy as np
from datetime import datetime

DEFAULT_SERVER_PORT = 8888
DEFAULT_NODE_NAME = 'Controller'
ctx = zmq.Context()


def run_server(port, name):
    print('[MESSAGING] Starting ZMQ listener... name:{}, port:{}'.format(name, port))

    sock = ctx.socket(zmq.REP)
    sock.bind('tcp://*:{}'.format(port))
    while True:
        data = sock.recv()
        msg = data.decode()
        th = threading.Thread(target=handle_message, args=(msg,))
        th.start()
        ack_msg = '{}->{}'.format(name, 'ack')
        sock.send(ack_msg.encode())

def run_client(target_ip, port):
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://{}:{}'.format(target_ip, port))
    print('[MESSAGING] client connected to {}:{}'.format(target_ip, port))
    return sock

def handle_message(msg):
    #print('[MESSAGING] listener received: {}'.format(msg))
    msg_dict = json.loads(msg)
    if msg_dict['type'] == 'img':
        img = unjsonify(msg_dict['img_string'])
        #cv2.imwrite(msg_dict['time']+'.jpg', img)
        #print(' - saved img.')
        print(msg_dict['time'])
        print(sys.getsizeof(img))

def unjsonify(msg):
    return np.array(msg)

def send_ctrl_msg(sock, msg):
    sock.send_string(msg)
    rep = sock.recv().decode()
    print(' - Reply from server: {}'.format(rep))


def create_message_list_numpy(nplist):
#    curTime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    curTime = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    json_msg = {'type': 'img', 'img_string': nplist, 'time' : curTime}
    return json.dumps(json_msg)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port, name = sys.argv[1].split(',', 1)
        th_listen = threading.Thread(target=run_server, args=(port, name))
        th_listen.start()
    else:
        th_listen = threading.Thread(target=run_server, args=(DEFAULT_SERVER_PORT, DEFAULT_NODE_NAME))
        th_listen.start()
        # run_server(DEFAULT_SERVER_PORT, DEFAULT_NODE_NAME)
