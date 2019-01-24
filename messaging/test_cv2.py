import cv2
import pickle
import sys
import threading
import base64
import message_bus


def load_image(img_file):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    frame = cv2.imread(img_file)
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    return data


def get_image_base64encoded(img_file):
    with open(img_file, "rb") as imageFile:
        str = base64.b64encode(imageFile.read())
        return str


if __name__ == '__main__':
    if len(sys.argv) > 1:
        port, name = sys.argv[1].split(',', 1)
    th_listen = threading.Thread(target=message_bus.run_server, args=(port, name))
    th_listen.start()

    while True:
        print('Type cmd to send: IP/PORT/type/payload')
        print(' - e.g. 143.248.55.122/8888/img/aa.jpg')
        print(' - e.g. 143.248.55.122/8888/text/abcdef')

        cmd = input()
        target_ip, target_port, cmd_type, msg = cmd.split('/', 3)
        target_sock = message_bus.run_client(target_ip, target_port)

        if cmd_type == 'img':
            #img_str_data = load_image(msg)
            img_str_data = get_image_base64encoded(msg)
            message_bus.send_ctrl_msg(target_sock, img_str_data)
        elif cmd_type == 'text':
            message_bus.send_ctrl_msg(target_sock, msg)
