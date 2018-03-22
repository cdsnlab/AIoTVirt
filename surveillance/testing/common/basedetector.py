# import the necessary packages
import cv2


class BaseDetector:
	def __init__(self, cascadePath):
		# load the face detector
		self.cascade = cv2.CascadeClassifier(cascadePath)

	def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
		# detect faces in the image
		rects = self.cascade.detectMultiScale(image,
			scaleFactor=scaleFactor, minNeighbors=minNeighbors,
			minSize=minSize, flags=cv2.CASCADE_SCALE_IMAGE)

		# return the rectangles representing bounding
		# boxes around the faces
		return rects
