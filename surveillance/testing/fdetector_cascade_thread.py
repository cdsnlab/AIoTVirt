from common.cascadedetector import CascadeDetector
from imutils.video import VideoStream
from datetime import datetime
import argparse
import imutils
import prctl
import time
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", default="../resources/cascades/haarcascade_frontalface_default.xml",
	help="path to where the face cascade resides")
args = ap.parse_args()

# Initialize the camera and grab a reference to the raw camera capture
# WARNING: VideoStream doesn't originally accept a "name" parameter. See
# the README in the  misc directory or simply remove the "name" kwarg for
# it to work normally
videostream = VideoStream(usePiCamera=True, name="CameraThread").start()

# Construct the face detector and allow the camrea to warm up
fd = CascadeDetector(args.face)
time.sleep(0.1)

# Set the thread's name
prctl.set_name("Cascade(Main)")

# Start the timer for FPS approximation
start = datetime.now()
frame_count = 0
# Capture frames from the camera
while True:
	# Grab the raw NumPy array representing the image
	frame = videostream.read()

	# Resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image and then clone the frame
	# so that we can draw on it
	faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5,
		minSize=(30, 30))
	frameClone = frame.copy()

	# Loop over the face bounding boxes and draw them
	for (fX, fY, fW, fH) in faceRects:
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

	# Show our detected faces, then clear the frame in
	# preparation for the next frame
	cv2.imshow("Face", frameClone)

	# If the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

	# Approximate FPS
	frame_count += 1
	time_elapsed = (datetime.now() - start).total_seconds()
	if time_elapsed >= 1:
		print("Approximate FPS: {0:.2f}".format(frame_count / time_elapsed), end="\r")
		frame_count = 0
		start = datetime.now()
