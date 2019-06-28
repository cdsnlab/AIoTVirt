# run this with python option not python3
# this automatically capture video when there is an object in view for given period of time in sync with other cameras

import os
import argparse
import datetime
import time
import subprocess
import socket
import imutils
import numpy as np
import cv2
import signal
import sys


class Rec(object):
    def __init__ (self):
        self.recordtime = None
        self.outputfilename = None
        self.out = None
        self.cap = None
        #self.commandline = "ffmpeg -f v4l2 -input_format mjpeg -i /dev/video0 -preset faster -pix_fmt yuv420p"
        self.cmd = ['ffmpeg', '-f', 'v4l2', '-input_format', 'mjpeg', '-i', '/dev/video0', '-preset', 'faster', '-pix_fmt', 'yuv422p']
        self.resolution = None
        self.min_area = None
        self.inputfps = 0
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self,sig,frame):
        print("BYBYBY")
        self.out.release()
        self.cap.release()
        sys.exit(0)

    def createcommandline(self):
        print('adding arguments...')
        fr = "-framerate" # frame rate
        ct = "-t" # capture time
        res = "-video_size" #resolution 
        bitrate = "-b:v" # bitrate
        ct_conv = str(datetime.timedelta(seconds=self.recordtime)) # converted time
        if(self.resolution == 1080):
            res_conv = "1920x1080"
            subprocess.call(['v4l2-ctl', '--set-fmt-video=width=1920,height=1080,pixelformat=YUV422p'])
        elif(self.resolution == 720):
            res_conv = "1280x720"
            subprocess.call(['v4l2-ctl', '--set-fmt-video=width=1280,height=720,pixelformat=YUV422p'])
        elif(self.resolution == 480):
            res_conv = "858x480"
            subprocess.call(['v4l2-ctl', '--set-fmt-video=width=858,height=480,pixelformat=YUV422p'])
        elif(self.resolution == 360):
            res_conv = "480x360"
            subprocess.call(['v4l2-ctl', '--set-fmt-video=width=480,height=360,pixelformat=YUV422p'])
        else:
            print ("invalid option")
        
        self.startrecord(fr, ct, res, ct_conv, res_conv, bitrate)

    def startrecord(self, fr, ct, res, ct_conv, res_conv, bitrate):
        print('running ffmpeg script...')
        self.cmd.append(bitrate)
        self.cmd.append(self.br_val)
        self.cmd.append(fr)
        self.cmd.append(str(self.inputfps))
        self.cmd.append(ct)
        self.cmd.append(ct_conv)
        self.cmd.append(res)
        self.cmd.append(res_conv)

        self.outputfilename = self.createoutfilename()
        self.cmd.append(self.outputfilename)
        print (self.cmd)
        subprocess.call(self.cmd)
        #print(self.commandline +" "+ fr +" "+ str(self.inputfps) +" "+ ct +" "+ ct_conv +" "+ res +" "+ res_conv +" "+ self.outputfilename)
        #os.system(self.commandline +" "+ fr +" "+ str(self.inputfps) +" "+ ct +" "+ ct_conv +" "+ res +" "+ res_conv +" "+ self.outputfilename)
            
    def createoutfilename(self):
        devicename = socket.gethostname()
        now = datetime.datetime.now()
        strnow = now.strftime("%H:%M")
        finalname = str(devicename) + "_" + strnow.replace(":", "") + "_" + str(self.resolution) + "_" + str(self.recordtime) +".mp4"
        print(finalname)
        return finalname
    
    def createoutfilename2(self):
        devicename = socket.gethostname()
        now = datetime.datetime.now()
        strnow = now.strftime("%H:%M")
        finalname = str(devicename) + "_" + strnow.replace(":", "") + "_" + str(self.recordtime)+".avi"
        return finalname

    def run(self):
        # read frames.
        occupied = False
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = self.createoutfilename2()
        print(filename)
#        self.out = cv2.VideoWriter(filename, 0x7634706d, 15.0, (720, 640))
#        self.out = cv2.VideoWriter(filename, 0x7634706d, 15.0, (1280, 720))
        self.out = cv2.VideoWriter(filename, fourcc, 10.0, (1280, 720))
        framecnt = 0
        firstFrame = None
        cum_fps = 0.0
        start_time = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame is None: #escape if no frames are grabbed.
                break
            if occupied:
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.out.write(frame)
                print(time.time() - begin)
                if (time.time() - begin) > self.recordtime :
                    print("stopping...")
                    occupied =False
                    time.sleep(1)
                    self.out.release()
            print(cum_fps, occupied, framecnt)
            #frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21),0)

            if firstFrame is None: #initialize first frame (nonmoving scene)
                firstFrame = gray
                continue

            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold (frameDelta, 45, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations = 2)
            cnts = cv2.findContours (thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours (cnts)
            print("number of contours: ", len(cnts))
            for c in cnts: # there is movement, start recording
                if cv2.contourArea(c) < self.min_area:
                    continue
                (x,y,w,h) = cv2.boundingRect(c)
                cv2.rectangle (frame, (x,y), (x+w, y+h), (0,255,0),2)
                #cv2.imwrite(str(framecnt)+".jpg", frame)
                # start record timer
                if(occupied != True):
                    begin = time.time()
                    print("starting...")
                    occupied = True
                    filename = self.createoutfilename2()
                    #self.out = cv2.VideoWriter (filename, 0x7634706d, 15.0, (720, 640))
                    #self.out = cv2.VideoWriter (filename, 0x7634706d, 15.0, (1280, 720))
                    #self.out = cv2.VideoWriter (filename, fourcc, 15.0, (720, 640))
                    self.out = cv2.VideoWriter(filename, fourcc, 10.0, (1280, 720))


            framecnt +=1    
            end_time = time.time()
            cum_fps = framecnt / (end_time - start_time)


        # if yes then start recording + msg other cams.

        # if no then stay in loop.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="recording test videos.")
    parser.add_argument('-t', '--recordtime', type=float, default = 30.0, help = 'length of the recording video in seconds')
    parser.add_argument('-o', '--output', type=str, default = 'cam_vid.mkv', help = 'name of the output file')
    parser.add_argument('-fps', '--inputfps', type=int, default = 30, help = ' frame rate of the input video')
    parser.add_argument('-r', '--resolution', type=int, default = 720, help = ' video resolution. type in vertical metric 720 or 480 (e.g. 1080p --> 1920x1080, 720p --> 1280x720, 480p --> 858x480, 360p --> 480x360)')
    parser.add_argument('-b', '--bitrate_value', type=str, default = "1M", help = ' desired bitrate. ')
    parser.add_argument('-ma', '--min_area', type=int, default = 500, help = ' minimum area for movement detection')
    
    ARGS = parser.parse_args()
    rec = Rec()
    rec.recordtime = ARGS.recordtime
    rec.outputfilename = ARGS.output
    rec.inputfps = ARGS.inputfps
    rec.resolution = ARGS.resolution
    rec.br_val = ARGS.bitrate_value
    rec.min_area = ARGS.min_area
    rec.run()
#    rec.createcommandline()


