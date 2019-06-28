# run this with python option not python3
# this capture video for given period of time

import os
import argparse
import datetime
import time
import subprocess
import socket
import zmq
import requests
import json
import sys

class Rec(object):
    def __init__ (self):
        self.recordtime = None
        self.cmd = ['ffmpeg', '-f', 'v4l2', '-input_format', 'mjpeg', '-i', '/dev/video0', '-preset', 'faster', '-pix_fmt', 'yuv422p']
        self.cmd_org = ['ffmpeg', '-f', 'v4l2', '-input_format', 'mjpeg', '-i', '/dev/video0', '-preset', 'faster', '-pix_fmt', 'yuv422p']
        self.resolution = None
        self.inputfps = 0
        self.writepwd = None
        self.ctx = None
        self.runtime = 14400.0
        self.cooltime = 10

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
        cmd = []
        cmd.clear()
        cmd = self.cmd_org.copy() # if you don't copy this list, u get smth messy
        print("1: ",cmd)
        cmd.append(bitrate)
        cmd.append(self.br_val)
        cmd.append(fr)
        cmd.append(str(self.inputfps))
        cmd.append(ct)
        cmd.append(ct_conv)
        cmd.append(res)
        cmd.append(res_conv)

        outputfilename = self.createoutfilename()
        cmd.append(outputfilename)
        print ("2: ",cmd)
        subprocess.call(cmd)
        if self.role == "server":
            #wait for server to do smth
            print("[INFO] finished recording, waiting for client to do smth")
        else: 
            print("[INFO] finished recording, sleeping for ", self.cooltime)
            time.sleep(self.cooltime)
            
    def createoutfilename(self):
        devicename = socket.gethostname()
        now = datetime.datetime.now()
        strnow = now.strftime("%H:%M")
        finalname = str(self.writepwd) + "/" + str(devicename) + "_" + strnow.replace(":", "") + "_" + str(self.resolution) + "_" + str(self.recordtime) +".mp4"
        return finalname

    def getcurrentdate(self):
        today = datetime.datetime.now()
        d4 = today.strftime("%m-%d-%Y") # prints Month-date-year
        return d4

    def run(self):
        if self.role == "client":
            sock = self.ctx.socket(zmq.REQ)
            sock.connect(f'tcp://192.168.1.213:7777')
            starttime = time.time()
        else: # server
            sock = self.ctx.socket(zmq.REP)
            sock.bind('tcp://*:7777')
        while True:
            # check todays date.
            mdy = self.getcurrentdate()
            pwd = "/home/cdsn/spencer/chameleon/utility/videoset/"+mdy
            if not os.path.exists (pwd): # if mdy exist, use it, otherwise create one.
                os.makedirs(pwd)
            self.writepwd =pwd
            if self.role == "client":
                loopingtime = time.time()
                print("[INFO] duration: ", loopingtime - starttime)
                if loopingtime - starttime > self.runtime:
                    print("[INFO] done for today, byebye")
                    byemsg = "bye"
                    sock.send(byemsg.encode())
                    byerecv = sock.recv(0)
                    if byerecv.decode()=="byebye":
                        self.slacknoti("end recording")
                        sys.exit(0)
                line = "start"
                sock.send(line.encode())
                print("[INFO] Sent start msg")
                recvmsg = sock.recv()
                if recvmsg.decode() == "ack":
                    print("[INFO] recv ack msg")
                    self.createcommandline()
                else:
                    print("[INFO] undefined control msg, quitting")
                    sys.exit(0)
            else: # server
                try:
                    time.sleep(1)
                    print("[INFO] waiting for clients message")
                    msg = sock.recv(0)
                    if msg.decode() == "start":
                        ackmsg = "ack"
                        sock.send(ackmsg.encode())
                        self.createcommandline()
                    elif msg.decode() == "bye":
                        byebyemsg = "byebye"
                        sock.send(byebyemsg.encode())
                        self.slacknoti("end recording")
                        sys.exit(0)

                    else:
                        print("error, exisiting")
                        sys.exit(0)

                except zmq.Again as e:
                    print("[INFO] no msg received yet")

    def slacknoti(self, contentstr):
        webhook_url = "https://hooks.slack.com/services/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"
        payload = {"text": contentstr}
        requests.post(webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="recording test videos.")
    parser.add_argument('-t', '--recordtime', type=int, default = 180, help = 'length of the recording video in seconds')
    parser.add_argument('-fps', '--inputfps', type=int, default = 30, help = ' frame rate of the input video')
    parser.add_argument('-r', '--resolution', type=int, default = 720, help = ' video resolution. type in vertical metric 720 or 480 (e.g. 1080p --> 1920x1080, 720p --> 1280x720, 480p --> 858x480, 360p --> 480x360)')
    parser.add_argument('-b', '--bitrate_value', type=str, default = "10M", help = ' desired bitrate. ')
    parser.add_argument('-ro', '--role', type=str, default = "client", help = ' desired role, client or server ')
    
    ARGS = parser.parse_args()
    rec = Rec()
    rec.recordtime = ARGS.recordtime
    rec.inputfps = ARGS.inputfps
    rec.resolution = ARGS.resolution
    rec.br_val = ARGS.bitrate_value
    rec.role = ARGS.role 
    rec.ctx = zmq.Context()
    rec.run()

