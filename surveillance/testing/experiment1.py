from common.detector import CascadeDetector
from common.detector import HOGDetector
from common.framegetter import TCPReq
from common.logger import Logger
from datetime import datetime
from time import sleep
import argparse
import imutils
# import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--video", "-v", help="path to the video file", type=str)
parser.add_argument("--descriptor", help="path to cascade descriptor",
					default="../resources/cascades/haarcascade_frontalcatface.xml", type=str)
parser.add_argument("--detector", choices=["hog", "cascade"], default="cascade", help="type of detector")
parser.add_argument("--scale", "-s", help="Scale Factor", default=1.1, type=float)
parser.add_argument("--winstride", "-w", help="Window Stride", default=(4, 4), type=tuple)
parser.add_argument("--display", "-d", action="store_true",
					help="Set flag to display frame on screen")
parser.add_argument("--log", nargs="?", const="experiment1", default=None,
					help="Wether to write the log or not")
parser.add_argument("--fps", help="Emulated FPS of video", type=float, default=20)
parser.add_argument("--host", help="DNS hostname", type=str, default="raspberrypi2")
args = parser.parse_args()

req = TCPReq(hostname=args.host)

if args.detector == "hog":
	detector = HOGDetector(scaleFactor=args.scale, winStride=args.winstride)
elif args.detector == "cascade":
	detector = CascadeDetector(args.descriptor)

if args.log is not None:
	logger = Logger(args.log)
	# The new logger logs all of the "registered" functions. In this case a simple
	# line stating the frame number and the number of objects detected
	msg = "{0:.2f}, {1}"
	# logger.register(lambda frame, detected, **kw: msg.format(frame, detected), "accuracy.csv")
	logger.register(lambda fps, breceived, **kw: msg.format(fps, breceived), "experiment1_meas.csv")
	# logger.register(lambda bstreamed: str(bstreamed))
	logger.register(logger.log_cpu)

# videosim = LiveVideoSim(video=args.video, scaleFactor=args.scale, winStride=args.winstride)

run_loop = True
fps = args.fps
shutter_time = 1 / fps
start = datetime.now()
extra_reads = 0
# The loop will add "1" to the counter and we want to start at the 0th frame
frame_ctr = -1
width = 300

try:
	while run_loop:
		# Get a frame from camera thread
		# t0 = datetime.now()
		frame = req.request_frame()
		# t1 = datetime.now()
		# print("Req time: {0}".format((t1 - t0).total_seconds()))
		if frame is None:
			break
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
			res = detector.display_frame(objRects)
			if not res:
				run_loop = False
		# Approximate FPS (returns fps if at least 1sec has passed since last update, else None)
		fps = detector.approx_fps()
		if fps:
			kw = {"fps": fps, "breceived": req.breceived,
				"frame": frame_ctr, "detected": len(objRects)}
			req.breceived = 0
			if args.log:
				logger.write_log(kw)
		# t4 = datetime.now()
		# print("Display/Stats Time: {0}".format((t4 - t3).total_seconds()))
		# sleep(1)
except KeyboardInterrupt:
	print("\nClosing application")
except Exception as e:
	raise e
	run_loop = False
finally:
	req.close()
	if args.log is not None:
		logger.close()
