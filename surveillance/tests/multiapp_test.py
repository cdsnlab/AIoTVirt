from multiprocessing import Process, Pipe, Value
from common.facedetector import FaceDetector
from imutils.video import VideoStream
from datetime import datetime
from functools import partial
import subprocess
import argparse
import imutils
import prctl
import time
import cv2


WAIT_INTERVAL = 0.001
FRAME_WIDTH = Value("i", 250)


def cascadeDetector(conn, width, descriptor, inter=None, name="", scaleFactor=1.1,
					minNeighbors=5, minSize=(35, 35), log=False):
	# Set process' name for it to be more easily identifiable with ps
	prctl.set_name("cdsn_" + name + "Det")

	# Open the log file for this process
	if log:
		logName = name + "Det_stats"
		f = open(logName, "a+")

	fd = FaceDetector(descriptor)
	time.sleep(0.1)

	# Start the timer for FPS approximation
	start = datetime.now()
	frame_count = 0
	try:
		# Capture frames from the camera
		while True:
			# Request the frame to the main process and wait for it
			conn.send(("get", width.value))
			frame = conn.recv()

			# ~~~Resize the frame and convert it to grayscale~~~
			# Now it is the main thread is doing this to avoid several processes
			# doing the same job
			# frame = imutils.resize(frame, width=300)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Detect faces in the image and then clone the frame
			# so that we can draw on it
			faceRects = fd.detect(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
									minSize=minSize)
			frameClone = frame.copy()

			# Loop over the face bounding boxes and draw them
			for (fX, fY, fW, fH) in faceRects:
				cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

			# Show our detected faces, then clear the frame in
			# preparation for the next frame
			cv2.imshow(name, frameClone)

			# If the 'q' key is pressed, stop the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

			# Approximate FPS
			frame_count += 1
			time_elapsed = (datetime.now() - start).total_seconds()
			if time_elapsed >= 1:
				fps = frame_count / time_elapsed
				print("Approximate FPS: {0:.2f}".format(fps), end="\r")
				frame_count = 0
				start = datetime.now()
				# Log data if desired (one might need to manually delete
				# previously created log files)
				if log:
					# We need to manually write linebreaks
					f.write(str(FRAME_WIDTH.value) + "\n")
					f.write(str(fps) + "\n")
					# cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py >> " + logName

					# We might search for "cdsn_..:" instead of 'py' but
					# I'll leave ot like this for now"
					cmd = "ps -eL -o comm,cmd,psr,pcpu | grep py"
					p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
					# get output of previously executed command
					out, err = p.communicate()
					f.write(out.decode())
					# Write separating empty newline
					f.write("\n")
	except Exception as e:
		print(e)
	finally:
		if log:
			f.close()


parser = argparse.ArgumentParser()
parser.add_argument("--show", "-s", action="store_true", help="set this flag to display the video")
parser.add_argument("--log", "-l", action="store_true", help="set this flag to log stats")
parser.add_argument("--interval", "-i", type=int, default=20,
					help="interval in seconds after which the frame size will change (if log is set)")
args = parser.parse_args()

FACE_DESCR_PATH = "../resources/cascades/haarcascade_frontalface_default.xml"
EYE_DESCR_PATH = "../resources/cascades/haarcascade_eye.xml"

faceDetector = partial(cascadeDetector, inter=args.interval, name="Face", descriptor=FACE_DESCR_PATH,
						log=args.log)
eyeDetector = partial(cascadeDetector, inter=args.interval, name="Eye", minSize=(20, 20),
						descriptor=EYE_DESCR_PATH, log=args.log)

# Start camera videostream
# WARNING: VideoStream doesn't originally accept a "name" parameter. See
# the README in the  misc directory or simply remove the "name" kwarg for
# it to work normally
videostream = VideoStream(usePiCamera=True, name="CameraThread").start()
time.sleep(0.1)

# Set the thread's name
prctl.set_name("Cascade(Main)")

# Initialise conection objects and processes
parentConn1, childConn1 = Pipe()
parentConn2, childConn2 = Pipe()
# p1 = Process(target=cascadeDetector, args=(childConn1, DESCR_PATH, "Cascade1"))
# p2 = Process(target=cascadeDetector, args=(childConn2, "cascades/haarcascade_eye.xml", "Cascade2"))
p1 = Process(target=faceDetector, args=(childConn1, FRAME_WIDTH))
p2 = Process(target=eyeDetector, args=(childConn2, FRAME_WIDTH))

p1.start()
p2.start()

t = datetime.now()

# Poll for frame requests in a Round Robin fashion. If there is a request
# serve it immediately
try:
	while True:
		if parentConn1.poll(WAIT_INTERVAL):
			recv, size = parentConn1.recv()
			if recv == "get":
				frame = videostream.read()
				frame = imutils.resize(frame, width=size)
				parentConn1.send(frame)
		if parentConn2.poll(WAIT_INTERVAL):
			recv, size = parentConn2.recv()
			if recv == "get":
				frame = videostream.read()
				frame = imutils.resize(frame, width=size)
				parentConn2.send(frame)
		if args.log and FRAME_WIDTH.value < 501:
			delta = (datetime.now() - t).total_seconds()
			if delta > args.interval:
				# print("timer reached")
				with FRAME_WIDTH.get_lock():
					FRAME_WIDTH.value += 50
				t = datetime.now()


except KeyboardInterrupt:
	parentConn1.close()
	parentConn2.close()
	childConn1.close()
	childConn2.close()
