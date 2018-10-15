import cv2
import numpy as np
from time import sleep

FACE_DESCR_PATH = "../resources/cascades/haarcascade_frontalface_default.xml"
BODY_DESCR_PATH = "../resources/cascades/haarcascade_fullbody.xml"
face_cascade = cv2.CascadeClassifier(BODY_DESCR_PATH)
FRAME_WIDTH =640 

vidcap = cv2.VideoCapture('../sampleVideo/midvideo.mp4')
#success,image = vidcap.read()
count =0
#success = True
while True:
    ### make files if you need to 
    success,image = vidcap.read()
    if success is False:
        break
    # cv2.imwrite("frame%d.jpg" % count, image)

    ### now detect!
    resize = cv2.resize(image, (480,360))
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    if count > 800:
        if count < 900:
            cv2.imwrite("tempImageFolders/frame_mid_new_%d.jpg" % count, resize)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(resize,(x,y),(x+w,y+h),(0,255,0),2)
        if faces is not None:
            print("faces detected at ", count) 
#            cv2.imwrite("frame%d.jpg" % count, resize)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = image[y:y+h, x:x+w]
#    cv2.imshow('image',resize)
    if cv2.waitKey(25) & 0xFF ==ord('q'):
        break
#    sleep(0.5)

    #success,image=vidcap.read()

    count+=1

vidcap.release()
cv2.destoryAllWindows()
