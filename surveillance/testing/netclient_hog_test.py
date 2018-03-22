from common.detector import HOGDetector
from common.framegetter import TCPReq
from common.logger import Logger
import argparse


SERVER_PORT = 5000
# BUFF_SIZE = 8192
FRAME_WIDTH = 300


class NetClient(TCPReq, HOGDetector, Logger):
	def __init__(self, **kw):
		super().__init__(**kw)


parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", default="jpg_compress",
					choices=["np_compress", "jpg_compress", "png_compress", "pickle"],
					help="method to send the objects")
parser.add_argument("--display", "-s", action="store_true", help="set this flag to display the video")
parser.add_argument("--log", "-l", action="store_true", help="set this flag to log stats")
args = parser.parse_args()

print("Frame-transmission compression method: {0}".format(args.method))
print("Display: {0}".format(args.display))
print("Starting detection algorithm")
faceDetector = NetClient(width=FRAME_WIDTH, method=args.method, port=SERVER_PORT,
						name="Face", log=args.log, display=args.display)

faceDetector.run()
