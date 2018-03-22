# from common.cascadedetector import CascadeDetector
from common.detector import CascadeDetector
from common.framegetter import PipeReq
from common.logger import Logger
from multiprocessing import Process, Value
from imutils.video import VideoStream
from aioprocessing import AioPipe
from contextlib import suppress
import argparse
import asyncio
import imutils
import prctl
import time


WAIT_INTERVAL = 0.001
FRAME_WIDTH = Value("i", 250)


class FaceDetector(PipeReq, CascadeDetector, Logger):
	def __init__(self, **kw):
		super().__init__(**kw)


class EyeDetector(PipeReq, CascadeDetector, Logger):
	def __init__(self, **kw):
		super().__init__(**kw)


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


faceDetector = FaceDetector(conn=childConn1, width=FRAME_WIDTH, descriptor=FACE_DESCR_PATH, name="Face",
								log=args.log, display=args.display)

eyeDetector = EyeDetector(conn=childConn2, width=FRAME_WIDTH, descriptor=EYE_DESCR_PATH, name="Eye",
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
