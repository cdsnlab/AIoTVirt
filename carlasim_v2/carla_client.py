import glob
import os
import sys

try:
   sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')[0])
except IndexError:
   pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
import numpy as np
import cv2

class ClientSideBoundingBoxes(object):
   """
   This is a module responsible for creating 3D bounding boxes and drawing them
   client-side on pygame surface.
   """

   @staticmethod
   def get_bounding_boxes(vehicles, camera):
      """
      Creates 3D bounding boxes based on carla vehicle list and camera.
      """

      bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
      # filter objects behind camera
      bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
      return bounding_boxes

   @staticmethod
   def draw_point(image, pos):
      return cv2.circle(cv2.UMat(image), pos, 10, (0,0,255), -1)
      # return image