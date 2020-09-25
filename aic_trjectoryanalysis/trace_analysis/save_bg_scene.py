'''
[DONE]
this program saves background scene of each video.
'''

import os, sys, time
import cv2
import numpy as np
from tqdm import tqdm
allfiles=[]

for (path, dir, files) in os.walk ("/home/spencer1/samplevideo/AIC20_track3_MTMC/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == ".avi" and "vdo" in os.path.splitext(filename)[0]:
            #print("%s/%s" % (path, filename))
            allfiles.append(path+"/"+filename)

def get_camera_id(filename):
    print(filename)
    items = filename.split('/')
    for item in items:
        if item.startswith("c0"):
            return item

for file in tqdm(allfiles):
    count=0
    cap = cv2.VideoCapture(file)

    camid = get_camera_id(file)
    while(count == 0):
        ret, frame = cap.read()
        cv2.imwrite('scene/'+camid+'.jpg',frame)
        count+=1