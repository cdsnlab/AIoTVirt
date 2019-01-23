import paho.mqtt.client as mqtt
import os

broker = "143.248.53.143"
port = 1883
timelive = 6000

def on_connect(client, userdata, flags, rc):
    print("connect with result code "+str(rc))

def on_message(client, userdata, msg):
    print(msg.payload.decode())
    # run a downloaded container. 
    #os.system("sudo docker run -it busybox &")
    
    # re/start an exited container. the container should already be created beforehand. 
    os.system("sudo docker restart test3")
    os.system("sudo docker exec -it test3 date +%T.%N")
    

def pub_message (client, topic, msg):
    print ("sending this..." + topic)
    ret = client.publish(topic, msg)

def sub_topic (client, topic):
    print ("subscribing to ..." + topic)
    ret = client.subscribe(topic)

client = mqtt.Client()
client.connect(broker,port,timelive)
client.on_connect = on_connect
client.on_message = on_message
client.pub_message = pub_message
client.loop_start()

if __name__ == '__main__':
#    main 돌리고... 
    main()   


