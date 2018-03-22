from imutils.video import VideoStream
from numpy import savez_compressed
from datetime import datetime
import argparse
import imutils
import asyncio
import time
import cv2
import io
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


class SocketVideoStreamer(asyncio.Protocol):
	"""Serve frames to the clients"""

	# asyncio already implements the generation of sockets, receiving etc, we
	# just have to provide the functions to react to certain events
	def __init__(self, videostream):
		self.videostream = videostream
		# self.method = method

	def connection_made(self, transport):
		# Print information to easily see if connections are arriving
		name = transport.get_extra_info("peername")
		print("Incoming connection from host {0}".format(name))
		# The transport is basically the socket object wrapped inside asyncio
		self.transport = transport

	def data_received(self, data):
		try:
			# The client sends msgs in the form "GET,<frame_width>"
			msg, size, method = data.decode().split(",")
			msg = msg.upper()
			size = int(size)
			# Sanity check and maybe we cann add other msgs if needed
			if msg == "GET":
				# Grab frame and send it to the client
				frame = self.videostream.read()
				frame = imutils.resize(frame, width=size)
				# Instantiate the buffer
				file = io.BytesIO()
				if method == "np_compress":
					# Save Zipped numpy array to the buffer. The function takes
					# the arrays to be saved as kwargs, in this case we are
					# saving the image in the variable _frame_ under the kwarg
					# "frame"
					savez_compressed(file, frame=frame)
					# Place pointer at the beginning of the buffer
					file.seek(0)
					out = file.read()
					# Send the frame followed by the "END" to signalize the end
					# of the transmission
					self.transport.write(out)
					self.transport.write("END".encode())
				elif method == "jpg_compress":
					# Quality 95 is actually the std value but I put it there
					# just to be able to play around with the parameter in case
					# it's needed.
					img_data = cv2.imencode(".jpg", frame,
											[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
					buff = io.BytesIO(img_data)
					out = buff.read()
					self.transport.write(out)
					self.transport.write("END".encode())
				else:
					pass
		except Exception as e:
			print("Error while processing received request")
			print(data)
			raise e


videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
time.sleep(0.1)

loop = asyncio.get_event_loop()
# Create the server based on our protocol
coro = loop.create_server(lambda: SocketVideoStreamer(videostream),
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
