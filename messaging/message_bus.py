import zmq
import threading
import base64
import cv2
import json
from node_table import NodeTable
import netif_util
import collections
from datetime import datetime,timedelta
import ntplib
from time import ctime

#
# Decorator for threading methods in a class
#
def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class MessageBus(object):
    def __init__(self, device_name, listen_port, role):
        self.ctx = zmq.Context()
        self.device_name = device_name
        self.listen_port = listen_port
        self.role = role
        self.handlers = collections.defaultdict(set)
        self.node_table = NodeTable()
        self.run_message_listener()

    #
    # Creates a socket for listening messages from other nodes
    # (both controllers and cameras).
    #
    @threaded
    def run_message_listener(self):
        print('[MESSAGING] Starting ZMQ listener... device_name:{}, port:{}'.format(self.device_name, self.listen_port))

        sock = self.ctx.socket(zmq.REP)
        sock.bind('tcp://*:{}'.format(self.listen_port))
        while True:
            data = sock.recv()
            msg = data.decode()
            self.handle_message(msg)
            ack_msg = '{}.{}'.format(self.device_name, 'ack')
            sock.send(ack_msg.encode())

    #
    # Sends a JSON Object through the ZeroMQ socket
    #
    def send_message_json(self, target_ip, target_port, msg_dict):
        sock = self.ctx.socket(zmq.REQ)
        sock.connect('tcp://{}:{}'.format(target_ip, target_port))
        # print('[MESSAGING] sending msg to {}:{} (type: {})'.format(target_ip, target_port, msg_dict['type']))
        sock.send_json(msg_dict)
        rep = sock.recv().decode()
        # print(' - Reply from receiver: {}'.format(rep))

    #
    # Sends a plain-text string through the ZeroMQ socket
    #
    def send_message_str(self, target_ip, target_port, msg_str):
        sock = self.ctx.socket(zmq.REQ)
        sock.connect('tcp://{}:{}'.format(target_ip, target_port))
        # print('[MESSAGING] sending msg to {}:{}'.format(target_ip, target_port))
        sock.send_string(msg_str)
        rep = sock.recv().decode()
        # print(' - Reply from receiver: {}'.format(rep))

    #
    # Retrieves an IPv4 address and a port number for
    # listening messages.
    #
    def get_my_node_info(self, iface_name):
        netif_list = netif_util.get_netif_list()
        for netif_item in netif_list:
            if netif_item['name'] == iface_name:
                return {'ip': netif_item['ipv4'], 'port': self.listen_port}
        return None

    #
    # Registers a callback function for handling received messages.
    # The parameter includes a message dictionary.
    #
    def register_callback(self, msg_type, callback):
        self.handlers[msg_type].add(callback)

    @threaded
    def handle_message(self, msg):
        try:
            msg_dict = json.loads(msg)

            if 'type' in msg_dict:
                if msg_dict['type'] == 'join':
                    # Reply to the client with the list of all joined nodes
                    for handler in self.handlers.get('join', []):
                        handler(msg_dict)
                elif msg_dict['type'] == 'device_list':
                    for handler in self.handlers.get('device_list', []):
                        handler(msg_dict)
                elif msg_dict['type'] == 'img':
                    for handler in self.handlers.get('img', []):
                        handler(msg_dict)
                elif msg_dict['type'] == 'img_metadata':
                    for handler in self.handlers.get('img_metadata', []):
                        handler(msg_dict)
                elif msg_dict['type'] == 'migration_request':
                    for handler in self.handlers.get('migration_request', []):
                        handler(msg_dict)
                else:
                    pass
            else:
                # Key 'type' does not exist. Discard the message.
                pass

        except json.decoder.JSONDecodeError:
            print(' - Error: invalid JSON format.')
            return
        except Exception as e:
            print(' - Error: general unknown error.')
            print(str(e))
            return

    @staticmethod
    def create_message_list_numpy(img, framecnt, encode_param, device_name,timegap=timedelta()):
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        encstr = base64.b64encode(encimg).decode('ascii')
        now = datetime.utcnow()+timegap
        curTime = now.strftime('%H:%M:%S.%f') # string format
        json_msg = {'type': 'img', 'img_string': encstr, 'time': curTime, 'framecnt': framecnt, 'device_name': device_name}
        return json.dumps(json_msg)
