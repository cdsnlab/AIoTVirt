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

    def read_frame(self):
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]

        image_stream = io.BytesIO()
        image_stream.write(self.connection.read(image_len))
        image_stream.seek(0)

        buff = np.fromstring(image_stream.getvalue(), dtype=np.uint8)
        cvImage = cv2.imdecode(buff, 1)

        return cvImage

    def monitor_frame(self, min_area=500):
        isDetected = False
        num = 0
        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied
            # text
            frame = self.read_frame()

            # resize the frame, convert it to grayscale, and blur it
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                # (x, y, w, h) = cv2.boundingRect(c)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # text = "Occupied"
                isDetected = True
                break
            if isDetected:
                self.profile_frame(frame)
                isDetected = False

    def profile_frame(self, frame):
        num = 0
        while True:
            if num != 0:
                frame = self.read_frame()
            faceCascade = cv2.CascadeClassifier("./cascade_file/haarcascade_profileface.xml")
            #           faceCascade = cv2.CascadeClassifier("./cascade_file/lbpcascade_frontalface_improved.xml")
            bodyCascade = cv2.CascadeClassifier("./cascade_file/haarcascade_upperbody.xml")
            accuracy = 0.0
            speed = time.time()
            faceNum, faceImage = self.detect_faces(faceCascade, frame, scaleFactor=1.1)
            bodyNum, bodyImage = self.detect_faces(bodyCascade, faceImage, scaleFactor=1.1, color=(255, 0, 0))

            # print("found")
            print("Found {0} faces {0} bodies!".format(faceNum, bodyNum))

            img = 'result_' + str(num)
            if (faceNum > 0):
                img += '_yes'
            else:
                img += '_no'
            if (bodyNum > 0):
                img += '_yes.jpg'
                accuracy += bodyNum
            else:
                img += '_no.jpg'
            cv2.imwrite(img, bodyImage)

            if num%self.duration == self.duration-1:
                speed = time.time()-speed
                accuracy = accuracy/self.duration*100.0
                if accuracy == 0.0:
                    break
                else:
                    self.send_profile(speed, accuracy)

            # cv2.imshow("image", cvImage)

    def send_profile(self, speed, accuracy):
        payload = {'node': self.name, 'name': 'DetectionSpeed', 'value': speed,
                   'updateTime': int(round(time.time() * 1000)), }
        publish.single("capability/CriminalTracking/DetectionSpeed", json.dumps(payload), hostname=self.ip,
                       port=self.port)

        payload = {'node': self.name, 'name': 'DetectionAccuracy', 'value': accuracy,
                   'updateTime': int(round(time.time() * 1000)), }
        publish.single("capability/CriminalTracking/DetectionAccuracy", json.dumps(payload),
                       hostname=self.ip,
                       port=self.port)