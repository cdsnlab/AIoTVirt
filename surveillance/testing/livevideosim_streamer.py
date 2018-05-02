from common.framegetter import FileVideoStreamLoop
from common.framegetter import VideoReq
from common.streamer import TCPStreamer
from common.logger import Logger
from contextlib import suppress
from datetime import datetime
# from math import floor
# from time import sleep
import argparse
import asyncio
# import imutils
# import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--video", "-v", help="path to the video file", type=str, required=True)
parser.add_argument("--scale", "-s", help="Scale Factor", default=1.1, type=float)
parser.add_argument("--winstride", "-w", help="Window Stride", default=(4, 4), type=tuple)
parser.add_argument("--display", "-d", action="store_true",
					help="Set flag to display frame on screen")
parser.add_argument("--fps", help="Emulated FPS of video", type=float, default=20)
args = parser.parse_args()

# req = VideoReq(video=args.video, resize=False)
# detector = HOGDetector(scaleFactor=args.scale, winStride=args.winstride)
logger = Logger("streamer")

# The new logger logs all of the "registered" functions. In this case a simple
# line stating the frame number and the number of objects detected
logger.register(lambda bsent, **kw: str(bsent), logfile="bw.log")
# logger.register(lambda frame, detected: str(frame) + "," + str(detected))
# logger.register(logger.log_cpu)

run_loop = True
fps = args.fps
shutter_time = 1 / fps
start = datetime.now()
extra_reads = 0
# The loop will add "1" to the counter and we want to start at the 0th frame
frame_ctr = -1
width = 300

ADDRESS = "0.0.0.0"
PORT = 5000
interval = 1


async def log(interval):
	while True:
		try:
			data = {"bsent": streamer.bytes_sent}
			logger.write_log(data)
			streamer.bytes_sent = 0
			await asyncio.sleep(interval)
		except asyncio.CancelledError:
			# break
			raise


video = args.video
videostream = FileVideoStreamLoop(video, queueSize=20, loop=True).start()
streamer = TCPStreamer(videostream)
# server = TCPStreamer(videostream=videostream)

loop = asyncio.get_event_loop()
try:
	# Start the coroutines
	coro = loop.create_server(lambda: TCPStreamer(videostream), ADDRESS, PORT)
	# coro = loop.create_server(streamer, ADDRESS, PORT)
	asyncio.ensure_future(log(interval))
	asyncio.ensure_future(coro)
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
	logger.close()
	# req.close()
