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

    @staticmethod
    def _create_bb_points(vehicle):
      """
      Returns 3D bounding box for a vehicle.
      """

      cords = np.zeros((8, 4))
      extent = vehicle.bounding_box.extent
      cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
      cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
      cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
      cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
      cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
      cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
      cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
      cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
      return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    # Handle video recording
    @staticmethod
    def process_img(image, VIEW_HEIGHT, VIEW_WIDTH):
        i = np.array(image.raw_data)
        i2 = i.reshape((VIEW_WIDTH, VIEW_HEIGHT, 4))
        i3 = i2[:, :, :3]
        return i3