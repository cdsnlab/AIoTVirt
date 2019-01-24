import sys, zmq
import time
import threading

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
        ack_msg = '{}->{}{}'.format(name, msg, 'ack')
        sock.send(ack_msg.encode())


def handle_message(msg):
    print('[MESSAGING] listener received: {}'.format(msg))
    print(' - start handling msg {}'.format(msg))
    time.sleep(5)
    print(' - handling msg {} complete'.format(msg))


def run_client(target_ip, port):
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://{}:{}'.format(target_ip, port))
    print('[MESSAGING] client connected to {}:{}'.format(target_ip, port))
    return sock


def send_ctrl_msg(sock, msg):
    print('[MESSAGING] client sending msg: {}'.format(msg))
    sock.send(msg.encode())
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
