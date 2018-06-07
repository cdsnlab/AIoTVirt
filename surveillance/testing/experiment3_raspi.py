from common.detector import CascadeDetector
from common.detector import HOGDetector
from contextlib import suppress
from datetime import datetime
from math import ceil
# import pandas as pd
import numpy as np
import argparse
import asyncio
import logging
import imutils
# import socket
import struct
import sys
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Show extra information")
parser.add_argument("--show-only", action="store_true",
					help="Show the images that would be sent without actually sending them and then exit")
parser.add_argument("--img", "-i", default="../resources/testImages/Starnberg.jpg")
parser.add_argument("--descriptor", help="path to cascade descriptor",
					default="../resources/cascades/haarcascade_frontalface_default.xml", type=str)
parser.add_argument("--detector", choices=["hog", "cascade"], default="cascade", help="type of detector")
parser.add_argument("--scale", "-s", help="Scale Factor", default=1.1, type=float)
parser.add_argument("--winstride", "-w", help="Window Stride", default=(4, 4), type=tuple)
parser.add_argument("--display", "-d", action="store_true",
					help="Set flag to display frame on screen")
parser.add_argument("--ip", help="Destination IP ",
					default="0.0.0.0")
parser.add_argument("-p",
					"--dest-port",
					help="Destination port to send the data to. "
						"(Default: 5000)",
					default=5000,
					type=int)
args = parser.parse_args()

if args.debug:
	level = logging.DEBUG
else:
	level = logging.INFO
logging.basicConfig(level=level)

DATAGRAM_PAYLOAD_SIZE = 512
dtypes = {"IMG": 0, "TEXT": 1}


class ConnManager(asyncio.Protocol):
	"""Handle TCP commands from the controller"""

	def __init__(self, queue, stop_ack_flag):
		self.stop_ack_flag = stop_ack_flag
		self.queue = queue
		# TODO: It might be even better to differentiate in the send_command
		# function between the different commands and in case of stop await
		# until it's done... But I don't know how to write an awaitable
		# function. Will look into it later
		asyncio.ensure_future(self.stop_ack())
		# self.command_list = ["start", "stop"]

	def connection_made(self, transport):
		# Print information to easily see if connections are arriving
		# This is the server's address
		logging.info("Manager connected")
		name = transport.get_extra_info("peername")
		logging.info("Succesfull connection to host {0}".format(name))
		# The transport is basically the socket object wrapped inside asyncio
		self.transport = transport

	def data_received(self, data):
		"""Data received are commands/messages from the controller"""
		logging.debug("Manager received '{}'".format(data))
		cmd = data.decode()
		asyncio.ensure_future(self.send_command(cmd))

	async def send_command(self, cmd):
		try:
			# if cmd not in self.command_list:
			# 	logging.error("Command not recognized: {}".format(cmd))
			# logging.info("ConnManager received command '{}'".format(cmd))

			# Send the received command to the UDPStreamer
			await self.queue.put(cmd)
		except asyncio.CancelledError:
			raise

	async def stop_ack(self):
		"""Notify the controller we are stopping operations"""
		try:
			while True:
				# Wait for the UDPStreamer to stop
				await self.stop_ack_flag.wait()
				logging.debug("ConnManager: Sending stop ACK")
				# Reset the flag
				self.stop_ack_flag.clear()
				# Send ACK to the controller
				self.transport.write("stop ACK".upper().encode())
		except asyncio.CancelledError:
			raise


class UDPStreamer(asyncio.DatagramProtocol):
	"""docstring for UDPStreamer"""

	def __init__(self, queue, stop_ack_flag, img_data):
		self.stop_ack_flag = stop_ack_flag
		self.img_data = img_data
		self.queue = queue
		self.run = False
		self._ready = asyncio.Event()
		self._start = asyncio.Event()
		# We have to be able to react to ConnManager's commands even between
		# sends so we make _await_command a separate coroutine
		asyncio.ensure_future(self._await_command())

	def connection_made(self, transport):
		# Print information to easily see if connections are arriving
		logging.info("UDP Streamer started")
		# The transport is basically the socket object wrapped inside asyncio
		self.transport = transport
		self._ready.set()

	def connection_lost(self, exc):
		logging.info('The server closed the connection')
		logging.warning("This is the end of the program but I don't know how to exit cleanly... Please pres ctrl c ;)")
		# I guess it's not the cleanest way to exit but well...
		# raise KeyboardInterrupt

	async def _await_command(self):
		try:
			# Wait until the transport has finished initialisation
			await self._ready.wait()
			while True:
				logging.debug("Awaiting for commands...")
				# ConnManager will send commands through the queue
				cmd = await self.queue.get()
				logging.debug("UDPStreamer received command '{}'".format(cmd))
				if cmd.startswith("start"):
					try:
						cmd, dtype = cmd.split(",")
					except ValueError:
						logging.error("Couldn't interpret command: {}".format(cmd))
						continue
					logging.info("Start received. Sending img data...")
					payload = self.img_data[dtype]
					self.run = True
					self._start.set()
					# We want to keep listening for commands so we send in a
					# separate coroutine
					asyncio.ensure_future(self.send_img_data(payload))
				elif cmd == "stop":
					logging.info("Stop received. Sending will halt.")
					# Set this flag so that the send coroutine breaks from the
					# loop
					self.run = False
					self.stop_ack_flag.set()
					# This flag might be unnecesary but anyway...
					self._start.clear()
				else:
					logging.error("[Error] Command not recognized {0}".format(cmd))
		except asyncio.CancelledError:
			raise

	async def send_img_data(self, payload):
		try:
			if not self._start.is_set():
				await self._start.wait()
			while True:
				if not self.run:
					break
				# Calculate how many datagrams we will need to send the image
				num_datagrams = ceil(len(payload) / DATAGRAM_PAYLOAD_SIZE)
				logging.debug("num datagrams: {}".format(num_datagrams))
				dtype = dtypes["IMG"]
				# The last datagram will most probably be smaller than the rest so send it
				# separately
				for seq in range(num_datagrams - 1):
					if self.run and len(payload) > 0:
						start_idx = seq * DATAGRAM_PAYLOAD_SIZE
						end_idx = (seq + 1) * DATAGRAM_PAYLOAD_SIZE
						fmt = "<BI{}s".format(DATAGRAM_PAYLOAD_SIZE)
						dgram_payload = struct.pack(fmt, dtype, seq, payload[start_idx:end_idx])
						self.transport.sendto(dgram_payload, (args.ip, args.dest_port))
						await asyncio.sleep(0)
					else:
						break
				else:
					seq = num_datagrams - 1
					start_idx = seq * DATAGRAM_PAYLOAD_SIZE
					dgram_payload = payload[start_idx:]
					# logging.debug("last package size: {0}. start idx: {1}. Seq {2}".format(len(dgram_payload), start_idx, seq))
					dgram_payload = struct.pack("<BI{0}s".format(len(dgram_payload)), dtype, seq, dgram_payload)
					if self.run:
						self.transport.sendto(dgram_payload, (args.ip, args.dest_port))
						await asyncio.sleep(0)
						self.transport.sendto("END".encode(), (args.ip, args.dest_port))
						await asyncio.sleep(0.001)
		except asyncio.CancelledError:
			self.transport.abort()
			raise

	# async def send_dgram(self, payload):
	# 	self.transport.write(payload)


if args.detector == "hog":
	detector = HOGDetector(scaleFactor=args.scale, winStride=args.winstride)
elif args.detector == "cascade":
	detector = CascadeDetector(args.descriptor)

# while run_loop:
# Get a frame from camera thread
# t0 = datetime.now()
frame = cv2.imread(args.img)
frame = imutils.resize(frame, width=300)
# t1 = datetime.now()
# print("Req time: {0}".format((t1 - t0).total_seconds()))
if frame is None:
	pass
	# break
# frame = imutils.resize(frame, width=width)
detector.frame = frame
# t2 = datetime.now()
# print("Alloc Time: {0}".format((t2 - t1).total_seconds()))
# Break if frammed couldn'tbe grabbed
# Get the box corners for the detected objects
objRects = detector.detect()
# t3 = datetime.now()
# print("Obj Det time: {0}".format((t3 - t2).total_seconds()))
# If desired, display the frame
if args.display:
	frameClone = frame.copy()
	for (fX, fY, fW, fH) in objRects:
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

	# Show our detected objects, then clear the frame in
	# preparation for the next frame
	# cv2.imshow("img", frameClone)

	# # If the 'q' key is pressed, stop the loop
	# cv2.waitKey(0)
img_data = cv2.imencode(".jpg", frame,
						[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
logging.debug("Uncompressed size: {0}".format(len(img_data)))

# I had problems when creating an image using np.zeros. With zeros_like it
# works as expected so I assume the error was due to data type
frameClone = np.zeros_like(frame)
t0 = datetime.now()
# Copy from the incoming frames the parts that we are interested in. The
# copy time was in the order of magnitude of 100 micro seconds so I think it
# wouldn't really affect performance since we are aiming at ~10fps which
# implies 100ms
cropped = None
for (fX, fY, fW, fH) in objRects:
	cropped = frame[fY: fY + fH, fX:fX + fW]
	frameClone[fY: fY + fH, fX:fX + fW] = cropped
# cropped = frame[fY: fY + fH, fX:fX + fW]
ellapsed = datetime.now() - t0
logging.debug("Time for copying ROI: {0}".format(ellapsed.total_seconds()))
# cv2.imshow("img", frameClone)
# # If the 'q' key is pressed, stop the loop
# cv2.waitKey(0)
# Compress the images
masked_data = cv2.imencode(".jpg", frameClone,
						[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring()
logging.debug("Masked size: {0}".format(len(masked_data)))
cropped_data = cv2.imencode(".jpg", cropped,
							[int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tostring() \
				if cropped is not None else masked_data
logging.debug("Compressed size: {0}".format(len(cropped_data)))
data_dict = {"full": img_data, "masked": masked_data, "cropped": cropped_data}

if args.show_only:
	cv2.imshow("img", frameClone)
	# # If the 'q' key is pressed, stop the loop
	# cv2.waitKey(0)
	cv2.imshow("img", frameClone)
	# # If the 'q' key is pressed, stop the loop
	# cv2.waitKey(0)
	cv2.imshow("img", frameClone)
	# # If the 'q' key is pressed, stop the loop
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	sys.exit(0)

SERVER_ADDRESS = args.ip
SERVER_PORT = args.dest_port

loop = asyncio.get_event_loop()
# queue = asyncio.Queue()
queue = asyncio.Queue(loop=loop)
stop_ack_flag = asyncio.Event()
try:
	# Create the server based on our protocol
	udp_coro = loop.create_datagram_endpoint(lambda: UDPStreamer(queue, stop_ack_flag, data_dict), remote_addr=(SERVER_ADDRESS, SERVER_PORT))
	tcp_coro = loop.create_connection(lambda: ConnManager(queue, stop_ack_flag), SERVER_ADDRESS, SERVER_PORT)
	udp_server = loop.run_until_complete(udp_coro)
	tcp_server = loop.run_until_complete(tcp_coro)
except ConnectionRefusedError:
	logging.error("Connection refused by server. It might be shutdown.")
	sys.exit(0)

try:
	loop.run_forever()
except KeyboardInterrupt:
	pass
except Exception as e:
	raise e

udp_server[0].close()
tcp_server[0].close()
pending = asyncio.Task.all_tasks()
for task in pending:
	task.cancel()
	# Now we should await task to execute it's cancellation.
	# Cancelled task raises asyncio.CancelledError that we can suppress:
	with suppress(asyncio.CancelledError):
		loop.run_until_complete(task)
loop.close()
