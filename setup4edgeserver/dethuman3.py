import numpy as numpy
import cv2
import argparse
import warnings
import json
import datetime
import imutils
import cv2
import time


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,help="path to the JSON configuration file")
args = vars(ap.parse_args())

warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

start_time = time.time()
vid = cv2.VideoCapture(conf["videofile"])
if(vid.isOpened()== False):
	print("error opening video stream or file")

haarfrontalface = cv2.CascadeClassifier(conf["haarfrontalface"])
haarfullbody = cv2.CascadeClassifier(conf["haarfullbody"])
read = 0

while (vid.isOpened()):
	ret, img = vid.read()
	if(ret!=False):
		break
	
	read+=1
	print("reading {0} frames".format(read))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = haarfrontalface.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	body = haarfullbody.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

vid.release()

elapsed_time = time.time() - start_time
print ("Found {0} faces!".format(len(faces)))
print ("Found {0} body!".format(len(body)))
print ("Read {0} frames!".format(len(read)))
print ("in {0} duration".format(elapsed_time))
print ("which means {0} fps".format(read/elapsed_time))
