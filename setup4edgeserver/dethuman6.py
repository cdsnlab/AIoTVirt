from PIL import Image
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
vid_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(conf["outputfile"],fourcc, 10.0, (int(vid_width),int(vid_height)))

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

        body = haarfullbody.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if len(body)!=0:
                bodycnt+=1
                print (body)
                for (x,y,w,h) in body:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        out.write(img)
                        #roi_gray = gray[y:y+h, x:x+w]
                        #roi_color = img[y:y+h, x:x+w]
                        #faces = haarfrontalface.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        #if len(faces)!=0:
                        #       facecnt+=1
                        #       for (bx,by,bw,bh) in faces:
                        #               cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
                                        

vid.release()
out.release()
elapsed_time = time.time() - start_time
#print ("Found {0} faces!".format(facecnt))
print ("Found {0} body!".format(bodycnt))
print ("Read {0} frames!".format(read))
print ("in {0} duration".format(elapsed_time))
print ("which means {0} fps".format(read/elapsed_time))
