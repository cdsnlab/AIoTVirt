from common.framegetter import VideoReq
from common.detector import HOGDetector
from common.logger import Logger
from datetime import datetime
from math import floor
from time import sleep
import argparse
# import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--scale", "-s", help="Scale Factor", default=1.1, type=float)
parser.add_argument("--winstride", "-w", help="Window Stride", default=(4, 4), type=tuple)
parser.add_argument("--video", "-v", help="patht to the video file", type=str)
parser.add_argument("--display", "-d", action="store_true",
					help="Set flag to display frame on screen")
parser.add_argument("--fps", help="Emulated FPS of video", type=float, default=20)
args = parser.parse_args()

# class LiveVideoSim(VideoReq, HOGDetector):
# 	"""docstring for LiveVideoSim"""
# 	def __init__(self, **kw):
# 		super().__init__(**kw)

req = VideoReq(video=args.video)
detector = HOGDetector(scaleFactor=args.scale, winStride=args.winstride)
logger = Logger()

# The new logger logs all of the "registered" functions. In this case a simple
# line stating the frame number and the number of objects detected
logger.register(lambda frame, detected: str(frame) + "," + str(detected))

# videosim = LiveVideoSim(video=args.video, scaleFactor=args.scale, winStride=args.winstride)

run_loop = True
fps = args.fps
shutter_time = 1 / fps
start = datetime.now()
extra_reads = 0
# The loop will add "1" to the counter and we want to start at the 0th frame
frame_ctr = -1

try:
	while run_loop:
		# Dummy reads to simulate lost frames
		for _ in range(extra_reads):
			req.request_frame()
		# Get a frame from camera thread
		detector.frame = req.request_frame()
		# Keep count of frames to compare with ground truth provided
		# frame_ctr = (frame_ctr + extra_reads + 1) % 10
		frame_ctr = frame_ctr + extra_reads + 1
		# Break if frammed couldn'tbe grabbed
		if detector.frame is None:
			break
		# Get the box corners for the detected objects
		objRects = detector.detect()
		# If desired, display the frame
		if args.display:
			res = detector.display_frame(objRects)
			if not res:
				run_loop = False
		# We have ground data for every 10th frame, we should log for these
		# frames to later calculate accuracy
		if frame_ctr % 10 == 0:
			logger.write_log(frame=frame_ctr, detected=len(objRects))
		# Approximate FPS
		detector.approx_fps()
		elapsed_time = (datetime.now() - start).total_seconds()
		# If we are processing too fast, sleep
		if elapsed_time < shutter_time:
			sleep(shutter_time - elapsed_time)
			extra_reads = 0
		# If we are processing too slow, calculate how many frames we have
		# missed
		else:
			extra_reads = floor(elapsed_time / shutter_time)
		# Restart the timer
		start = datetime.now()
except Exception as e:
	raise e
	run_loop = False
finally:
	req.close()
	logger.close()
