import configparser
import sys
sys.path.insert(0, '../messaging')
from message_bus import MessageBus
from datetime import datetime
import base64
import cv2
import numpy as np
import signal


class Controller(object):
    def __init__(self, name, port):
        self.msg_bus = MessageBus(name, port, 'controller')
        self.msg_bus.register_callback('join', self.handle_message)
        self.msg_bus.register_callback('img', self.handle_message)
        self.msg_bus.register_callback('img_metadata', self.handle_message)
        signal.signal(signal.SIGINT, self.signal_handler)
        self.logfile = None
 
    def signal_handler(self, sig, frame):
        self.logfile.close()
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
        # cv2.imwrite(msg_dict['time']+'.jpg', decimg)
        # print(' - saved img.')
        curTime = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3] # string format
        curdatetime = datetime.strptime(curTime, '%H:%M:%S.%f')
        sentdatetime = datetime.strptime(msg_dict['time'], '%H:%M:%S.%f')
        self.logfile.write(str(msg_dict['framecnt'])+"\t"+curTime+"\t"+str(sentdatetime - curdatetime)+"\t"+str(sys.getsizeof(decimg))+'\n')
        print(" - from: ", msg_dict['device_name'])
        print(" - timebtw: ", sentdatetime-curdatetime)
        print(" - size: ", sys.getsizeof(decimg))  # 0.3MB

    def process_image_metadata(self, msg_dict):
        print(' - image metadata')
        print(' - %s' % msg_dict)


if __name__ == '__main__':
    # Read 'master.ini'
    config = configparser.ConfigParser()
    config.read('../resource/config/master.ini')
    listen_port = config['controller']['port']
    controller_name = config['controller']['name']
    logfile_name = config['logs']['logfilename']
    ctrl = Controller(controller_name, listen_port)
    
    ctrl.logfile = open(logfile_name, 'w')

