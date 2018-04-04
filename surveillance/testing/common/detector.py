from imutils.object_detection import non_max_suppression
from functools import partial
from datetime import datetime
import numpy as np
import prctl
import cv2


class BaseDetector(object):
	"""docstring for BaseDetector"""

	def __init__(self, name="", display=False, **kw):
		super().__init__(**kw)
		self.display = display
		self.run_loop = True
		self.name = name
		prctl.set_name("cdsn_" + name + "Det")
		# Start the timer and frame count for FPS approximation
		self.frame_count = 0
		self.start = datetime.now()

	# def preprocessing(self):
	# 	pass

	# def postprocessing(self):
	# 	pass

	# def postprocessing(self):
	def approx_fps(self):
		"""Approximate fps (processing speed)"""
		self.frame_count += 1
		time_elapsed = (datetime.now() - self.start).total_seconds()
		if time_elapsed >= 1:
			fps = self.frame_count / time_elapsed
			print("Approximate FPS: {0:.2f}".format(fps), end="\r")
			self.frame_count = 0
			self.start = datetime.now()
			# Log data if desired (one might need to manually delete
			# previously created log files)
			# UPDATE: Logger is now a separate module
			# if self.log:
			# 	self.write_log(fps, self.width)

	# @deprecation.deprecated(details="User should write his own run loop")
	def run(self):
		# WARNING: You should be writing your own run loop. This one is kept for
		# compatibility with the examples I was testing with

		try:
			while self.run_loop:
				# Get a frame from camera thread
				self.frame = self.request_frame()
				# Break if frammed couldn'tbe grabbed
				if not self.frame:
					break
				# Get the box corners for the detected objects
				objRects = self.detect()
				# If desired, display the frame
				if self.display:
					res = self.display_frame(objRects)
					if not res:
						self.run_loop = False
				# Approximate FPS
				self.approx_fps()
		except Exception as e:
			print(e)
			self.run_loop = False
		finally:
			super().close()
			# self.conn.close()
			# if self.log:
			# 	self.f.close()


class CascadeDetector(BaseDetector):
	"""docstring for CascadeDetector"""

	def __init__(self, descriptor, scaleFactor=1.1, minNeighbors=5,
				minSize=(35, 35), **kw):
		super().__init__(**kw)
		classifier = cv2.CascadeClassifier(descriptor)
		# returns the rectangles representing bounding boxes around the faces
		self.det = partial(classifier.detectMultiScale, scaleFactor=scaleFactor,
							minNeighbors=minNeighbors, minSize=minSize,
							flags=cv2.CASCADE_SCALE_IMAGE)

	def detect(self):
		gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		objRects = self.det(gray)
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
			# self.run_loop = False
			return False
		return True


class HOGDetector(BaseDetector):
	"""docstring for HOGDetector"""

	def __init__(self, scaleFactor=1.1, winStride=(4, 4), padding=(8, 8), **kw):
		super().__init__(**kw)
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		self.det = partial(hog.detectMultiScale, winStride=winStride,
							padding=padding, scale=scaleFactor)

	def detect(self):
		(rects, weights) = self.det(self.frame)
		# print(len(rects))
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		# if not pick:
		# 	pick = []
		return pick

	def display_frame(self, objRects):
		# Clone the frame so that we can draw on it
		frameClone = self.frame.copy()
		# print(len(frameClone))
		# Loop over the object bounding boxes and draw them
		for (xA, yA, xB, yB) in objRects:
			cv2.rectangle(frameClone, (xA, yA), (xB, yB), (0, 255, 0), 2)

		# Show our detected objects, then clear the frame in
		# preparation for the next frame
		cv2.imshow(self.name, frameClone)

		# If the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			# self.run_loop = False
			return False
		return True
