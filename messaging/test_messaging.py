import message_bus
import sys
import threading

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port, name = sys.argv[1].split(',', 1)
    # port = 8888
    # name = 'Controller'
    th_listen = threading.Thread(target=message_bus.run_server, args=(port, name))
    th_listen.start()

    while True:
        print('Type cmd to send: IP/PORT/PAYLOAD')
        print(' - e.g. 143.248.55.122/8888/dfafsdafda')
        cmd = input()
        target_ip, target_port, payload = cmd.split('/', 2)
        target_sock = message_bus.run_client(target_ip, target_port)
        message_bus.send_ctrl_msg(target_sock, payload)
