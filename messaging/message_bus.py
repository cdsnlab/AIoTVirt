import sys, zmq
import time
import threading
import base64
import cv2
import json
import pickle


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


def decode_image(img_b64encoded):
    img_binary = base64.b64decode(img_b64encoded)
    fp = open('recv.jpg', 'wb')
    fp.write(img_binary)
    fp.close()
    #cv2.imshow('ImageWindow', img_binary)
    img = cv2.imread('recv.jpg', 0)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
    cv2.imshow("messsssi", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def handle_message(msg):
    print('[MESSAGING] listener received: {}'.format(msg))
    msg_dict = json.loads(msg)
    if msg_dict['type'] == 'img':
        decode_image(msg_dict['img_string'].encode('ascii'))
        print(' - saved img.')


def run_client(target_ip, port):
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://{}:{}'.format(target_ip, port))
    print('[MESSAGING] client connected to {}:{}'.format(target_ip, port))
    return sock


def send_ctrl_msg(sock, msg):
    print('[MESSAGING] client sending msg: {}'.format(msg))
    # sock.send(msg.encode())
    sock.send_string(msg)
    rep = sock.recv().decode()
    print(' - Reply from server: {}'.format(rep))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        port, name = sys.argv[1].split(',', 1)
        th_listen = threading.Thread(target=run_server, args=(port, name))
        th_listen.start()
    else:
        th_listen = threading.Thread(target=run_server, args=(DEFAULT_SERVER_PORT, DEFAULT_NODE_NAME))
        th_listen.start()
        # run_server(DEFAULT_SERVER_PORT, DEFAULT_NODE_NAME)
