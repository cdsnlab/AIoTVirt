from common.cascadedetector import CascadeDetector
from multiprocessing import Process, Value
from imutils.video import VideoStream
from aioprocessing import AioPipe
import paho.mqtt.client as mqtt
from contextlib import suppress
from datetime import datetime
import argparse
import asyncio
import imutils
import prctl
import json
import time


WAIT_INTERVAL = 0.001
FRAME_WIDTH = Value("i", 250)
SERVER_ADDR = "143.248.56.213"
SERVER_PORT = 18830
TOPIC_HEADER = "video/lab/"


# TODO: Refactor this function to be a Class with several methods for
# better code clarity. Might be a common Class for other scripts
class MqttPublisher(CascadeDetector):
	"""docstring for MqttPublisher"""

	def __init__(self, conn, width, descriptor, name="", scaleFactor=1.1,
				minNeighbors=5, minSize=(35, 35), log=False, display=False):
		super().__init__(conn, width, descriptor, name, scaleFactor,
						minNeighbors, minSize, log, display)
		self.client = mqtt.Client(self.name + "Det")
		self.client.connect(SERVER_ADDR, SERVER_PORT)
		self.client.on_connect = self.on_connect
		self.prev_state = {"Object": self.name, "Num": 0}

	# The callback for when the client receives a CONNACK response from the server.
	def on_connect(client, userdata, flags, rc):
		print("Connected with result code " + str(rc))
		# Subscribing in on_connect() means that if we lose the connection and
		# reconnect then subscriptions will be renewed.
		client.subscribe("$SYS/#")

	# def on_publish(client, userdata, mid):
	# 	print("")

	def run(self):
		# Start the timer and frame count for FPS approximation
		self.start = datetime.now()
		self.frame_count = 0
		try:
			while self.run_loop:
				# Get a frame from camera thread
				self.request_frame()
				# Get the box corners for the detected objects
				objRects = self.detect()
				# Current object count
				curr_count = len(objRects)
				# If there was a change in state, send it to the controller and
				# update the state
				if curr_count != self.prev_state["Num"]:
					self.prev_state["Num"] = curr_count
					res, mid = self.client.publish(TOPIC_HEADER + "cam1", json.dumps(self.prev_state))
					print(res)
					# self.client.publish(TOPIC_HEADER + "cam1", curr_count)
				# If desired, display the frame
				if self.display:
					self.display_frame(objRects)
				# Approximate FPS and log if desired
				self.approx_fps()
		except Exception as e:
			print(e)
			self.run_loop = False
		finally:
			if self.log:
				self.f.close()


async def serveFrames(parentConn):
	"""Listen for frame requests on the specified connection"""
	while True:
		try:
			recv, size = await parentConn.coro_recv()
			if recv == "get":
				frame = videostream.read()
				frame = imutils.resize(frame, width=size)
				parentConn.send(frame)
		except asyncio.CancelledError:
			# break
			raise


async def changeImgWidth(interval):
	"""Increase frame width at specified interval"""
	if args.log:
		while FRAME_WIDTH.value < 501:
			try:
				with FRAME_WIDTH.get_lock():
					FRAME_WIDTH.value += 50
				await asyncio.sleep(interval)
			except asyncio.CancelledError:
				# break
				raise


parser = argparse.ArgumentParser()
parser.add_argument("--display", "-s", action="store_true", help="set this flag to display the video")
parser.add_argument("--log", "-l", action="store_true", help="set this flag to log stats")
parser.add_argument("--interval", "-i", type=int, default=20,
					help="interval in seconds after which the frame size will change (if log is set)")
args = parser.parse_args()

FACE_DESCR_PATH = "../resources/cascades/haarcascade_frontalface_default.xml"
EYE_DESCR_PATH = "../resources/cascades/haarcascade_eye.xml"

# Start camera videostream
# WARNING: VideoStream doesn't originally accept a "name" parameter. See
# the README in the  misc directory or simply remove the "name" kwarg for
# it to work normally
videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
time.sleep(0.1)

# Set the thread's name
prctl.set_name("Cascade(Main)")

# Initialise conection objects and processes
parentConn1, childConn1 = AioPipe()
parentConn2, childConn2 = AioPipe()
faceDetector = MqttPublisher(childConn1, FRAME_WIDTH, FACE_DESCR_PATH, name="Face",
							log=args.log, display=args.display)
eyeDetector = MqttPublisher(childConn2, FRAME_WIDTH, EYE_DESCR_PATH, name="Eye",
							log=args.log, display=args.display, minSize=(20, 20))

p1 = Process(target=faceDetector.run)
p2 = Process(target=eyeDetector.run)

p1.start()
p2.start()

# Start asyncio loop
loop = asyncio.get_event_loop()
try:
	# Start the coroutines
	asyncio.ensure_future(serveFrames(parentConn1))
	asyncio.ensure_future(serveFrames(parentConn2))
	asyncio.ensure_future(changeImgWidth(args.interval))
	loop.run_forever()
except KeyboardInterrupt:
	# This indicates normal exit
	print("Stopping Application")
except Exception as e:
	# Print any unexpected exception
	print(e)
finally:
	# Tasks are still running at this point so we have to terminate them
	pending = asyncio.Task.all_tasks()
	for task in pending:
		task.cancel()
		# Wait for the cancellation of the task. Cancelled tasks raise
		# asyncio.CancelledError, which we can suppress
		with suppress(asyncio.CancelledError):
			loop.run_until_complete(task)
	# Close pipes
	parentConn1.close()
	parentConn2.close()
	childConn1.close()
	childConn2.close()
	# Stopping or closing the loop produces an error. Probably because it
	# doesn't have enough time to await for the tasks. So just let the
	# program finish by itself
	# loop.stop()
	# loop.close()
