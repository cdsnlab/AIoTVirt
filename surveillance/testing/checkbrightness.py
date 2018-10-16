import cv2
import numpy as np

def getBrightness(frame):
    bright = np.average(frame)
    return bright