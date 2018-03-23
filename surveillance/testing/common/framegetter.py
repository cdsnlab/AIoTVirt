from imutils.video import VideoStream
import numpy as np
import imutils
import socket
import time
import cv2
import io


class VideoReq(object):
	"""docstring for VideoReq"""

	def __init__(self, video, width=300, **kw):
		super().__init__(**kw)
		self.camera = cv2.VideoCapture(video)

	def request_frame(self):
		res, frame = self.camera.read()
		if res:
			frame = imutils.resize(frame, width=300)
			self.frame = frame
		return res


class StreamReq(object):
	"""docstring for LocalReq"""

	def __init__(self, width=300, **kw):
		super().__init__(**kw)
		# Initialize the camera and grab a reference to the raw camera capture
		# WARNING: VideoStream doesn't originally accept a "name" parameter. See
		# the README in the  misc directory or simply remove the "name" kwarg for
		# it to work normally
		videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
		time.sleep(0.1)
		self.videostream = videostream
		self.width = width

	def request_frame(self):
		frame = self.videostream.read()
		self.frame = imutils.resize(frame, width=300)
		return True


class PipeReq(object):
	"""docstring for PipeReq"""

	def __init__(self, conn, width=300, **kw):
		super().__init__(**kw)
		self.conn = conn
		self.width = width

	def request_frame(self):
		# Request the frame to the main process and wait for it
		self.conn.send(("get", self.width.value))
		self.frame = self.conn.recv()
		return True

	def close(self):
		self.conn.close()


class TCPReq(object):
	"""Get frames from camera node"""

	def __init__(self, method, hostname="raspberrypi", port=5000, width=300, **kw):
		super().__init__(**kw)
		# Get server addr through dns (I am assuming they are in the same network). If
		# we know the nodes' addr ahead we can also harcode them...
		print("Connecting to camera node")
		addr = socket.gethostbyname(hostname)
		soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		soc.connect((addr, port))
		self.conn = soc
		self.method = method
		self.width = width

	def request_frame(self):
		msg = "GET," + str(self.width) + "," + self.method
		self.conn.send(msg.encode())
		data = b""
		buff = self.conn.recv(8192)
		while len(buff):
			# print(type(buff), len(buff))
			try:
				# print(buff[-30:])
				if buff[-3:].decode() == "END":
					data += buff[:-3]
					break
			except UnicodeError:
				pass
			# if buff.decode() == "END":
			# 	break
			data += buff
			buff = self.conn.recv(8192)
			# print(type(data))
		# print(len(data))
		if self.method == "np_compress":
			self.frame = np.load(io.BytesIO(data))["frame"]
		elif self.method == "jpg_compress":
			data = np.fromstring(data, dtype="uint8")
			# self.frame = cv2.imdecode(data, int(cv2.LOAD_IMAGE_COLOR))
			# Flag > 0 indicates decoding as a colour image. (We implement
			# converting to gray in another step)
			self.frame = cv2.imdecode(data, 1)
		else:
			print("Method not implemented yet!")
			raise Exception("Method not implemented yet!")
		return True

	def close(self):
		self.conn.close()
