from common.basedetector import BaseDetector
from datetime import datetime
import subprocess
import prctl
import time
import cv2


class CascadeDetector(object):
	"""docstring for CascadeDetector"""

	def __init__(self, conn, width, descriptor, name="", scaleFactor=1.1,
				minNeighbors=5, minSize=(35, 35), log=False, display=False):
		self.conn = conn
		self.width = width
		self.name = name
		self.scaleFactor = scaleFactor
		self.minNeighbors = minNeighbors
		self.minSize = minSize
		self.log = log
		self.display = display
		self.run_loop = True

		# Set process' name for it to be more easily identifiable with ps
		prctl.set_name("cdsn_" + name + "Det")

		# Open the log file for this process
		if log:
			logName = name + "Det_stats"
			self.f = open(logName, "a+")

		self.bd = BaseDetector(descriptor)
		time.sleep(0.1)

	def request_frame(self):
		# Request the frame to the main process and wait for it
		self.conn.send(("get", self.width.value))
		self.frame = self.conn.recv()

	def detect(self):
		# ~~~Resize the frame and convert it to grayscale~~~
		# Now it is the main thread is doing this to avoid several processes
		# doing the same job
		# frame = imutils.resize(frame, width=300)
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

		# Detect objects in the image
		objRects = self.bd.detect(gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors,
								minSize=self.minSize)
		return objRects

	def display_frame(self, objRects):
		# Clone the frame so that we can draw on it
		frameClone = self.frame.copy()
		# Loop over the object bounding boxes and draw them
		for (fX, fY, fW, fH) in objRects:
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

		# Show our detected objects, then clear the frame in
		# preparation for the next frame
		cv2.imshow(self.name, frameClone)

		# If the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			self.run_loop = False

	def approx_fps(self):
		self.frame_count += 1
		time_elapsed = (datetime.now() - self.start).total_seconds()
		if time_elapsed >= 1:
			fps = self.frame_count / time_elapsed
			print("Approximate FPS: {0:.2f}".format(fps), end="\r")
			self.frame_count = 0
			self.start = datetime.now()
			# Log data if desired (one might need to manually delete
			# previously created log files)
			if self.log:
				self.write_log(fps)

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
				# If desired, display the frame
				if self.display:
					self.display_frame(objRects)
				# Approximate FPS
				self.approx_fps()
		except Exception as e:
			print(e)
			self.run_loop = False
		finally:
			self.conn.close()
			if self.log:
				self.f.close()

	def write_log(self, fps):
		# We need to manually write linebreaks
		self.f.write(str(self.width.value) + "\n")
		self.f.write(str(fps) + "\n")
		# cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py >> " + logName

		# We might search for "cdsn_..:" instead of 'py' but
		# I'll leave ot like this for now"
		cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py"
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
		# get output of previously executed command
		out, err = p.communicate()
		self.f.write(out.decode())
		# Write separating empty newline
		self.f.write("\n")
