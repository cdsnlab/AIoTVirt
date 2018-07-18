# this program send raw video stream to the edge server.

import socket
import time
import picamera

# Connect a client socket to my_server:8000 (change my_server to the
# hostname of your server)
client_socket = socket.socket()
# change the socket receive port for your use!
client_socket.connect(('143.248.53.143', 40909))

# Make a file-like object out of the connection
connection = client_socket.makefile('wb')
try:
     camera = picamera.PiCamera()
     camera.resolution = (1920, 1080)
     camera.framerate = 2
     # Start a preview and let the camera warm up for 2 seconds
     # camera.start_preview()
     time.sleep(2)
     # Start recording, sending the output to the connection for 60
     # seconds, then stop
     camera.start_recording(connection, format='h264')
     camera.wait_recording(60)
     camera.stop_recording()
finally:
     connection.close()
     client_socket.close()