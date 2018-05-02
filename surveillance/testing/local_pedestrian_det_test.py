from common.detector import HOGDetector
from common.framegetter import StreamReq
from common.framegetter import VideoReq
from common.logger import Logger
import argparse


FRAME_WIDTH = 300


class NetClient(StreamReq, HOGDetector, Logger):
	def __init__(self, **kw):
		super().__init__(**kw)


parser = argparse.ArgumentParser()
parser.add_argument("--video", "-v", help="path to the video")
parser.add_argument("--display", "-s", action="store_true", help="set this flag to display the video")
parser.add_argument("--log", "-l", action="store_true", help="set this flag to log stats")
args = parser.parse_args()

print("Starting detection algorithm")
pedestrianDetector = NetClient(name="Person", width=FRAME_WIDTH,
								log=args.log, display=args.display)

pedestrianDetector.run()
