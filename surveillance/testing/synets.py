from common.streamer import TCPStreamer
from imutils.video import VideoStream
# from numpy import savez_compressed
# from datetime import datetime
# import argparse
# import imutils
import asyncio
import time
# import cv2
# import io
# import socket


# We could also use 127.0.0.1 or localhost' but in my experience, on Debian it
# works most of the times better by putting 0.0.0.0 (all loopback addresses)
ADDRESS = "0.0.0.0"
PORT = 5000

# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind((ADDRESS, PORT))
# server.listen(2)

# while True:
# 	conn, addr = server.accept()
# 	msg = conn.recv(128)
# 	msg = msg.decode()
# 	if msg == "GET":
# 		# Grab Frame

# 		conn.send(frame)


videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
time.sleep(0.1)

loop = asyncio.get_event_loop()
# Create the server based on our protocol
coro = loop.create_server(lambda: TCPStreamer(videostream),
							ADDRESS, PORT)
server = loop.run_until_complete(coro)

print('Serving on {}'.format(server.sockets[0].getsockname()))
try:
	loop.run_forever()
except KeyboardInterrupt:
	pass

server.close()
loop.run_until_complete(server.wait_closed())
loop.close()
