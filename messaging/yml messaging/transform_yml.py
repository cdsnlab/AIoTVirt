import os, sys, time
import paho.mqtt.client as paho
import socket
import yaml
import subprocess
import ast

os.chdir(os.path.dirname(os.path.realpath(__file__)) ) #change cwd


broker = "143.248.53.10"

ip2name_table = {"143.248.1.1":"ctrl_1", "143.248.1.2":"robot","192.168.1.218":"cam1_1","143.248.57.73":"fogos_server", "143.248.53.69":"cam1_1", "127.0.1.1":"cam1_1"}


local_ip = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2]
if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)),
s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET,
socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

#local_ip = socket.gethostbyname(socket.gethostname())
print("my ip is "+local_ip)
my_name = ip2name_table[local_ip]


def on_message(client, userdata, message):
    print("received message =",str(message.payload.decode("utf-8")))
    payload_str=message.payload.decode("utf-8")
    payload = ast.literal_eval(payload_str)
    executable = payload["executable"]
    arguements = []
    for key in payload:
        if not(key =='executable') and not(key =='fogos_node') and not(key == 'replicas'):
            arguements.append('--'+key)
            arguements.append(str(payload[key]))
    running_command = ["python3", executable]+arguements
    print("running", running_command)
    subprocess.run(running_command)

def on_publish(client,userdata,result):             #create function for callback
    print("data published\n")
    pass

#client= paho.Client("matcher")
client= paho.Client()
client.on_message=on_message
client.on_publish=on_publish

print("connecting to broker ",broker)
client.connect(broker)#connect
topic = "fogos_node/"
client.subscribe(topic+my_name)
print("subscribed "+topic+my_name)

client.loop_start() 

while True:
    if os.path.exists("rm.yml"):
        with open("rm.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config['services']:
                if key == "controller":
                    msgtopic = topic + config['services'][key]['fogos_node']
                    payload = config['services'][key]
                    client.publish(msgtopic, str(payload))

                elif key == "robot":
                    msgtopic = topic + config['services'][key]['fogos_node']
                    payload = config['services'][key]
                    client.publish(msgtopic, str(payload))

                elif key == "camera":
                    for i in range(1, config['services'][key]['replicas']+1):
                        msgtopic = topic + config['services'][key]['fogos_node']+str(i)
                        payload = config['services'][key]
                        client.publish(msgtopic, str(payload))

                elif key == "fogos_server":
                    msgtopic = topic + config['services'][key]['fogos_node']
                    payload = config['services'][key]
                    client.publish(msgtopic, str(payload))


    #* delete the robot_monitoring.yml file after publishing the messages?!
    #* os.remove("robot_monitoring.yml")
        time.sleep(100)
