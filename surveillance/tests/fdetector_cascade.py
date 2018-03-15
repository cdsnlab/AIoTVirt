from common.facedetector import FaceDetector
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
	help="path to where the face cascade resides")
args = ap.parse_args()

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
# camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# construct the face detector and allow the camera to warm up
fd = FaceDetector(args.face)
time.sleep(0.1)

# start the timer for FPS approximation
start = datetime.now()
frame_count = 0
# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image
	frame = f.array

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the image and then clone the frame
	# so that we can draw on it
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5,
		minSize=(30, 30))
	frameClone = frame.copy()

	# loop over the face bounding boxes and draw them
	for (fX, fY, fW, fH) in faceRects:
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

	# show our detected faces, then clear the frame in
	# preparation for the next frame
	cv2.imshow("Face", frameClone)
	rawCapture.truncate(0)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	# Approximate FPS
	frame_count += 1
	time_elapsed = (datetime.now() - start).total_seconds()
	if time_elapsed >= 1:
		print("Approximate FPS: {0:.2f}".format(frame_count / time_elapsed), end="\r")
		frame_count = 0
		start = datetime.now()
