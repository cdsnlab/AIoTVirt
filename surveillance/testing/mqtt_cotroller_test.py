import paho.mqtt.client as mqtt
from datetime import datetime

SERVER_ADDR = "143.248.56.213"
SERVER_PORT = 18830


def on_message(client, userdata, msg):
	if msg.topic.find("SYS") == -1:
		print("{0} message received".format(datetime.now()))
		print("topic {0}: {1}".format(msg.topic, msg.payload.decode("utf-8")))
		# print("message qos:", msg.qos)
		# print("message retain flag:", msg.retain)


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
	print("Connected with result code " + str(rc))
	# Subscribing in on_connect() means that if we lose the connection and
	# reconnect then subscriptions will be renewed.
	client.subscribe("$SYS/#")


print("Starting mqtt client")
client = mqtt.Client("controller")
client.connect(SERVER_ADDR, SERVER_PORT)
client.subscribe("video/lab/#")
client.on_message = on_message
client.on_connect = on_connect
try:
	print("Waiting for messages")
	client.loop_forever()
except KeyboardInterrupt:
	print("Stopping Loop")
except Exception as e:
	raise e
finally:
	pass
	# client.loop_stop()
