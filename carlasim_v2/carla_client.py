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

    @staticmethod
    # ! Unnecessary
    def draw_bounding_boxes(image, bounding_boxes, VIEW_WIDTH, VIEW_HEIGHT):
        """
        Draws bounding boxes on pygame display.
        """
        # bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        # bb_surface.set_colorkey((0, 0, 0))
        image  = ClientSideBoundingBoxes.process_img(image, VIEW_WIDTH, VIEW_HEIGHT)
        # print(image.format)
        # cv2.imwrite('regular1.jpg', image)
        BB_COLOR = (0,0,255)
        line_thickness = 2
        image = cv2.UMat(image)
        for bbox in bounding_boxes:
            box = np.delete(bbox, 2, 1)
            # bbox = bbox[:4]
            point = box.mean(0).getA().astype(int)
            image = cv2.circle(image, (point[0][0], point[0][1]), 10, BB_COLOR, -1)
            # points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # print(points)
            # # draw lines
            # # base
            # image = cv2.line(image, points[0], points[1], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[0], points[1], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[1], points[2], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[2], points[3], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[3], points[0], BB_COLOR, line_thickness)
            # # top
            # image = cv2.line(image, points[4], points[5], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[5], points[6], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[6], points[7], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[7], points[4], BB_COLOR, line_thickness)
            # # base-top
            # image = cv2.line(image, points[0], points[4], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[1], points[5], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[2], points[6], BB_COLOR, line_thickness)
            # image = cv2.line(image, points[3], points[7], BB_COLOR, line_thickness)
        # cv2.imshow
        # cv2.imwrite('lines.jpg', image)
        return image

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox
