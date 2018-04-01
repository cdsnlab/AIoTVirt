import os
import io
import socket
import struct
import time
import cv2
import json
from PIL import Image
from sys import argv
import numpy as np
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

class Profiler:
    def __init__(self, duration):
        # create server socket
        self.client_socket = socket.socket()
        self.client_socket.connect(('localhost', 1234))
        self.connection = self.client_socket.makefile('rb')
        self.duration = duration

    def detect_faces(self, f_cascade, colored_img, scaleFactor = 1.1, color=(0,255,0)):
        img_copy = colored_img.copy()

        #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        #let's detect multiscale (some images may be closer to camera than others) images
        faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

        #go over list of faces and draw them as rectangles on original colored img
        #for (x, y, w, h) in faces:
        #    cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 2)

        return len(faces), img_copy

    def read_image(self):
        try:
            faceCascade = cv2.CascadeClassifier("./cascade_file/haarcascade_profileface.xml")
#           faceCascade = cv2.CascadeClassifier("./cascade_file/lbpcascade_frontalface_improved.xml")
            bodyCascade = cv2.CascadeClassifier("./cascade_file/haarcascade_upperbody.xml")
            accuracy = 0.0
            speed = time.time()
            for i in range(self.duration):
                self.image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

                self.image_stream = io.BytesIO()
                self.image_stream.write(self.connection.read(self.image_len))
                self.image_stream.seek(0)

                buff = np.fromstring(self.image_stream.getvalue(), dtype=np.uint8)
                cvImage = cv2.imdecode(buff, 1)

                faceNum, faceImage = self.detect_faces(faceCascade, cvImage, scaleFactor=1.1)
                bodyNum, bodyImage = self.detect_faces(bodyCascade, faceImage, scaleFactor=1.1, color=(255,0,0))

                #print("found")
                print("Found {0} faces {0} bodies!".format(faceNum,bodyNum))

                img = 'result_'+str(i)
                if( faceNum > 0 ):
                    img += '_yes'
                else:
                    img += '_no'
                if( bodyNum > 0 ):
                    img += '_yes.jpg'
                    accuracy += 1.0
                else:
                    img += '_no.jpg'
                cv2.imwrite(img, bodyImage)

                #cv2.imshow("image", cvImage)
        finally:
            speed = time.time() - speed
            accuracy = accuracy/self.duration*100

            self.connection.close()
            self.client_socket.close()
            return speed, accuracy