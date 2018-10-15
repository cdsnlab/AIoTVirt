#!/usr/local/bin/python3

import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='stitchedoutput.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = 'tempImageFolders/'
#ext = args['frame_new']
ext = 'frame_new'
output = args['output']

images = []
file_list = os.listdir(dir_path)
#load all 'frame_new' files.
for f in sorted(file_list):
#for f in sorted(file_list, key = lambda x:x[-6:]):
#    if f.startswith(ext):
     images.append(f)

# sort them 

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in images:

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    out.write(frame) # Write out frame to video

    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))

