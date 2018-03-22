from common.cascadedetector import CascadeDetector
from multiprocessing import Process, Pipe, Value
from imutils.video import VideoStream
# from functools import partial
from datetime import datetime
import argparse
import imutils
import prctl
import time


WAIT_INTERVAL = 0.001
FRAME_WIDTH = Value("i", 250)


parser = argparse.ArgumentParser()
parser.add_argument("--show", "-s", action="store_true", help="set this flag to display the video")
parser.add_argument("--log", "-l", action="store_true", help="set this flag to log stats")
parser.add_argument("--interval", "-i", type=int, default=20,
					help="interval in seconds after which the frame size will change (if log is set)")
args = parser.parse_args()

FACE_DESCR_PATH = "../resources/cascades/haarcascade_frontalface_default.xml"
EYE_DESCR_PATH = "../resources/cascades/haarcascade_eye.xml"

# The detector is now an object intead of a function
# faceDetector = partial(cascadeDetector, inter=args.interval, name="Face", descriptor=FACE_DESCR_PATH,
# 						log=args.log)
# eyeDetector = partial(cascadeDetector, inter=args.interval, name="Eye", minSize=(20, 20),
# 						descriptor=EYE_DESCR_PATH, log=args.log)

# Start camera videostream
# WARNING: VideoStream doesn't originally accept a "name" parameter. See
# the README in the  misc directory or simply remove the "name" kwarg for
# it to work normally
videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
time.sleep(0.1)

# Set the thread's name
prctl.set_name("Cascade(Main)")

# Initialise conection objects and processes
parentConn1, childConn1 = Pipe()
parentConn2, childConn2 = Pipe()

faceDetector = CascadeDetector(childConn1, FRAME_WIDTH, FACE_DESCR_PATH, name="Face",
								log=args.log, display=args.display)
eyeDetector = CascadeDetector(childConn2, FRAME_WIDTH, EYE_DESCR_PATH, name="Eye",
								log=args.log, display=args.display, minSize=(20, 20))

p1 = Process(target=faceDetector.run)
p2 = Process(target=eyeDetector.run)

p1.start()
p2.start()

t = datetime.now()

# Poll for frame requests in a Round Robin fashion. If there is a request
# serve it immediately
try:
	while True:
		if parentConn1.poll(WAIT_INTERVAL):
			recv, size = parentConn1.recv()
			if recv == "get":
				frame = videostream.read()
				frame = imutils.resize(frame, width=size)
				parentConn1.send(frame)
		if parentConn2.poll(WAIT_INTERVAL):
			recv, size = parentConn2.recv()
			if recv == "get":
				frame = videostream.read()
				frame = imutils.resize(frame, width=size)
				parentConn2.send(frame)
		if args.log and FRAME_WIDTH.value < 501:
			delta = (datetime.now() - t).total_seconds()
			if delta > args.interval:
				# print("timer reached")
				with FRAME_WIDTH.get_lock():
					FRAME_WIDTH.value += 50
				t = datetime.now()

except KeyboardInterrupt:
	parentConn1.close()
	parentConn2.close()
	childConn1.close()
	childConn2.close()
