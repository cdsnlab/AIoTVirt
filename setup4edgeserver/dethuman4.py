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
facecnt=0
bodycnt=0
while (vid.isOpened()):
	ret, img = vid.read()
	if(ret!=True): # there are no more image frames. 
		break
	read+=1
	#print("reading {0} frames".format(read))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = haarfrontalface.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	if len(faces)!=0:
		#print ("found faces")
		facecnt+=1
	body = haarfullbody.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
	if len(body)!=0:
		#print("found bodies")
		bodycnt+=1

vid.release()

elapsed_time = time.time() - start_time
print ("Found {0} faces!".format(facecnt))
print ("Found {0} body!".format(bodycnt))
print ("Read {0} frames!".format(read))
print ("in {0} duration".format(elapsed_time))
print ("which means {0} fps".format(read/elapsed_time))
