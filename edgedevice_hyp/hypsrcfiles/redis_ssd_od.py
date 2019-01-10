#!/usr/bin/python3

# ****************************************************************************
# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
# ****************************************************************************

# Detect objects on a LIVE camera feed using
# Intel® Movidius™ Neural Compute Stick (NCS)

import os
import cv2
import sys
import numpy
import ntpath
import argparse
import time

import mvnc.mvncapi as mvnc

import csv
import psutil

from utils import visualize_output
from utils import deserialize_output

from collections import deque
import numpy as np

import redis

# Detection threshold: Minimum confidance to tag as valid detection
CONFIDANCE_THRESHOLD = 0.80 # 60% confidant

# buffer for pltos 
PLOTBUFFSIZE = 5

# Variable to store commandline arguments
ARGS                 = None

# OpenCV object for video capture
camera               = None

# is it live video stream or from file?
live                 = 1
# estimated frame rate
fps                  = 0
# video file name
vidname              = ''

width                = 1
height               = 1

# elapsed time
#elapsedtime          = ''
inftime              = 0
starttime            = ''

# files opened
 
# log cpu usage
cpu                  =0.0
# log number of objects detected per frame.
numobj               = 0
# number of previous plots to determine 
plotbuff             = deque(maxlen=PLOTBUFFSIZE)
counter              = 0
direction            = ""
# ---- Step 1: Open the enumerated device and get a handle to it -------------

def open_ncs_device():

    # Look for enumerated NCS device(s); quit program if none found.
    devices = mvnc.EnumerateDevices()
    if len( devices ) == 0:
        print( "No devices found" )
        quit()

    # Get a handle to the first enumerated device and open it
    device = mvnc.Device( devices[0] )
    device.OpenDevice()

    return device

# ---- Step 2: Load a graph file onto the NCS device -------------------------

def load_graph( device ):

    # Read the graph file into a buffer
    with open( ARGS.graph, mode='rb' ) as f:
        blob = f.read()

    # Load the graph buffer into the NCS
    graph = device.AllocateGraph( blob )

    return graph

# ---- Step 3: Pre-process the images ----------------------------------------

def pre_process_image( frame ):

    # Resize image [Image size is defined by choosen network, during training]
    img = cv2.resize( frame, tuple( ARGS.dim ) )

    # Convert RGB to BGR [OpenCV reads image in BGR, some networks may need RGB]
    if( ARGS.colormode == "rgb" ):
        img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype( numpy.float16 )
    img = ( img - numpy.float16( ARGS.mean ) ) * ARGS.scale

    return img

# ---- Step 4: Read & print inference results from the NCS -------------------

def infer_image( graph, img, frame ):

    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd( 
                      output, 
                      CONFIDANCE_THRESHOLD, 
                      frame.shape )

    # Print the results (each image/frame may have multiple objects)
    print( "I found these objects in "
            + " ( %.2f ms ):" % ( numpy.sum( inference_time ) ) )

    for i in range( 0, output_dict['num_detections'] ):
        print( "%3.1f%%\t" % output_dict['detection_scores_' + str(i)] 
               + labels[ int(output_dict['detection_classes_' + str(i)]) ]
               + ": Top Left: " + str( output_dict['detection_boxes_' + str(i)][0] )
               + " Bottom Right: " + str( output_dict['detection_boxes_' + str(i)][1] ) )

        # Draw bounding boxes around valid detections 
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]

        # Prep string to overlay on the image
        display_str = ( 
                labels[output_dict.get('detection_classes_' + str(i))]
                + ": "
                + str( output_dict.get('detection_scores_' + str(i) ) )
                + "%" )

        frame = visualize_output.draw_bounding_box( 
                       y1, x1, y2, x2, 
                       frame,
                       thickness=4,
                       color=(255, 255, 0),
                       display_str=display_str )

    print( '\n' )

    # If a display is available, show the image on which inference was performed
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS live inference', frame )


# Spencer 4-1: added fps-------
def infer_image_fps( graph, img, frame, fps ):
#    global direction
    global counter
#    global plotbuff
    a = []
    # Load the image as a half-precision floating point array
    graph.LoadTensor( img, 'user object' )

    # Get the results from NCS
    output, userobj = graph.GetResult()

    # Get execution time
    inference_time = graph.GetGraphOption( mvnc.GraphOption.TIME_TAKEN )

    # Deserialize the output into a python dictionary
    output_dict = deserialize_output.ssd( 
                      output, 
                      CONFIDANCE_THRESHOLD, 
                      frame.shape )


#    elapsedtime = time.time() - starttime
    # Print the results (each image/frame may have multiple objects)
#    print( "I found these objects in ( %.2f ms ):" % ( numpy.sum( inference_time ) ) )
    inftime = numpy.sum(inference_time)
    numobj = (output_dict['num_detections'])
    
    # create array for detected obj
    a = [[] for _ in range(numobj)]

#    print (numobj)
    cpu = psutil.cpu_percent()
     
    for i in range( 0, output_dict['num_detections'] ):
        print( "%3.1f%%\t" % output_dict['detection_scores_' + str(i)] 
               + labels[ int(output_dict['detection_classes_' + str(i)]) ]
               + ": Top Left: " + str( output_dict['detection_boxes_' + str(i)][0] )
               + " Bottom Right: " + str( output_dict['detection_boxes_' + str(i)][1] ) )
#        print(str(i))
        a[i].append(output_dict['detection_scores_' + str(i)])
        a[i].append(labels[ int(output_dict['detection_classes_' + str(i)]) ])
        a[i].append(str(output_dict['detection_boxes_' + str(i)][0]))
        a[i].append(str(output_dict['detection_boxes_' + str(i)][1]))
        # Draw bounding boxes around valid detections 
        (y1, x1) = output_dict.get('detection_boxes_' + str(i))[0]
        (y2, x2) = output_dict.get('detection_boxes_' + str(i))[1]
 
        # Prep string to overlay on the image


        display_str = (labels[output_dict.get('detection_classes_' + str(i))] + ": " + str( output_dict.get('detection_scores_' + str(i) ) ) + "%" )
        
        frame = visualize_output.draw_bounding_box( 
                       y1, x1, y2, x2, 
                       frame,
                       thickness=4,
                       color=(255, 255, 0),
                       display_str=display_str )
        cv2.putText(frame, 'FPS:' + str(fps), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),2,cv2.LINE_AA)
#        cv2.putText(frame, direction, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255),3)
#    print( '\n' )

    # If a display is available, show the image on which inference was performed
    counter += 1
    if displayit == "on":
        cv2.imshow( 'NCS live inference', frame )

    # need to log here.
#    print (elapsedtime)
    # for tracking one human... 
#    print(str("{0:.2f}".format(elapsedtime)) + '\t' + str(cpu) + '\t' + str("{0:.2f}".format(inftime)) + '\t' + str("{0:.2f}".format(fps)) +'\t' + str(numobj)+'\t'+str(a)+'\n')
    
    # need to save to redis.
    save = {"elapsedtime": "{0:.2f}".format(elapsedtime), "CPU": str(cpu), "inftime": str("{0:.2f}".format(inftime)), "fps": str("{0:.2f}".format(fps)), "numberofobjects": str(numobj),"a": str(a)}

    r.hmset(counter, save)
#    r.sadd("myset",save)
    print(save)
    # need plots...! for multiple objects
    del(a)



# ---- Step 5: Unload the graph and close the device -------------------------

def close_ncs_device( device, graph ):
    graph.DeallocateGraph()
    device.CloseDevice()
    camera.release()
    cv2.destroyAllWindows()

# ---- Main function (entry point for this script ) --------------------------

def main():
    # open file for logging.
    global f
    global r
    global elapsedtime
    f = open(logfile, 'w', encoding = 'utf-8',newline='')
    r = redis.Redis(host='localhost', port=6379, db=0)
    prevTime = 0
    starttime = time.time()
    device = open_ncs_device()
    graph = load_graph( device )

    # Main loop: Capture live stream & send frames to NCS
    if live == 1:
        device = open_ncs_device()

        while( True ):
            ret, frame = camera.read()
            #### get fps
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            fps = 1/(sec)

            print("estimated live fps {0}" . format(fps))


        ####
            img = pre_process_image( frame )
#            infer_image (graph, img, frame)
        # this is spencers code for infering fps.
            infer_image_fps( graph, img, frame, fps )

        # Display the frame for 5ms, and close the window so that the next
        # frame can be displayed. Close the window if 'q' or 'Q' is pressed.
            if( cv2.waitKey( 1 ) & 0xFF == ord( 'q' ) ):
                break

        close_ncs_device( device, graph )
    # read video from file
    else:
        cap = cv2.VideoCapture(live)

        while (cap.isOpened()):
            ret, frame = cap.read()
            smallerimg = cv2.resize( frame, (width, height)) 
            
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            fps = 1/(sec)

            elapsedtime = time.time() - starttime
#            print(elapsedtime)
# fps, elapsedtime, inference time, cpu usage, num of object is ready. 

#            print("estimated file fps {0}" . format(fps))
            img = pre_process_image(smallerimg)
#            img = pre_process_image(frame)
#            infer_image (graph, img, frame)
            infer_image_fps(graph, img, smallerimg, fps)

#            f.write(str(elapsedtime) + '\t' + str(cpu) + '\t' + str(inftime) + '\t' + str(fps) +'\t' + str(numobj)+'\n')
#            infer_image_fps(graph,img,frame,fps)

            if(cv2.waitKey(3) & 0xFF == ord( 'q' )):
                break
        cap.release()
    f.close()
# ---- Define 'main' function as the entry point for this script -------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                         description="Detect objects on a LIVE camera feed using \
                         Intel® Movidius™ Neural Compute Stick." )

    parser.add_argument( '-g', '--graph', type=str,
                         default='../../caffe/SSD_MobileNet/graph',
                         help="Absolute path to the neural network graph file." )

    parser.add_argument( '-v', '--video', type=int,
                         default=0,
                         help="Index of your computer's V4L2 video device. \
                               ex. 0 for /dev/video0" )

    parser.add_argument( '-l', '--labels', type=str,
                         default='../../caffe/SSD_MobileNet/labels.txt',
                         help="Absolute path to labels file." )

    parser.add_argument( '-M', '--mean', type=float,
                         nargs='+',
                         default=[127.5, 127.5, 127.5],
                         help="',' delimited floating point values for image mean." )

    parser.add_argument( '-S', '--scale', type=float,
                         default=0.00789,
                         help="Absolute path to labels file." )

    parser.add_argument( '-D', '--dim', type=int,
                         nargs='+',
                         default=[300, 300],
                         help="Image dimensions. ex. -D 224 224" )

    parser.add_argument( '-c', '--colormode', type=str,
                         default="bgr",
                         help="RGB vs BGR color sequence. This is network dependent." )

    parser.add_argument( '-w', '--width', type=int,
                         default="600",
                         help="width of the capturing videos." )

    parser.add_argument( '-hi', '--height', type=int,
                         default="400",
                         help="height of the capturing videos." )

    parser.add_argument( '-vf', '--videofile', type=str,
                         default="1",
                         help="load from video file." )

    parser.add_argument( '-dis', '--display', type=str,
                         default="off",
                         help="load from video file." )

    ARGS = parser.parse_args()

    # Create a VideoCapture object
    camera = cv2.VideoCapture( ARGS.video )

    # Set camera resolution
    camera.set( cv2.CAP_PROP_FRAME_WIDTH, ARGS.width )
    camera.set( cv2.CAP_PROP_FRAME_HEIGHT, ARGS.height )
    width = ARGS.width
    height = ARGS.height
    # is it live?
    live = ARGS.videofile

    # name of the log file
    #logfile = ARGS.logfile
    logfile = ARGS.videofile + "_" + str(ARGS.width) + "_" + str(ARGS.height) + "_"+ str(CONFIDANCE_THRESHOLD) +"_" + ARGS.graph[12:18] + ".txt"

    # is display on?
    displayit = ARGS.display
    
    # Load the labels file
    labels =[ line.rstrip('\n') for line in
              open( ARGS.labels ) if line != 'classes\n']
    main()

# ==== End of file ===========================================================
