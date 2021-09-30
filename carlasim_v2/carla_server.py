import sys, glob
try:
    sys.path.append(glob.glob('../carla/dist/carla-0.9.11-py3.6-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

import configparser
import csv
import os
import queue
import random
import time
from collections import OrderedDict
import cv2
import numpy as np
import pygame

from client_carla import ClientSideBoundingBoxes

class SynchronousServer(object):
    def __init__(self, data_dir_path, config_file_path, port, record_mode, script_name):

        self._queues = []
        self.tracks = {}
        self.rgb_images = {}
        self.semseg_images = {}
        self.world = None
        self.frame = None
        self._settings = None
        self._queues = []
        self.pedestrians = []
        self.camera_ids = []
        self.num_of_tracks_done = 0
        self.record_mode = 0#record_mode
        self.scrip_name = script_name

        self.parse_cam_config(config_file_path)
        self.make_dirs(data_dir_path)
        self.connect_client(port)
        self.setup_camera()
    
        #self.image_map = {cam: set() for cam in range()}

    def parse_cam_config(self, config_file_path):
        print('[INFO] Parsing the configurations...')
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        
        self.cam_count = int(self.config['GENERAL']['Cameras'])

        self.view_width = int(self.config['GENERAL']['Image_Size_X'])
        self.view_height = int(self.config['GENERAL']['Image_Size_Y'])
        self.view_fov = int(self.config['GENERAL']['FOV'])
        self.fps = int(self.config['GENERAL']['FPS'])
        self.fixed_delta = 1 / self.fps


        self.rgb_camera_list = []
        self.semseg_camera_list = []
        print('[INFO] Configurations Parsed')

    def setup_camera(self):
        for cam_id in range(self.cam_count):
            camera_config = self.config['CAMERA_' + str(cam_id)]
            location = carla.Location(x=float(camera_config['PosX']), y=float(camera_config['PosY']), z=float(camera_config['PosZ']))
            rotation = carla.Rotation(roll=float(camera_config['Roll']), pitch=float(camera_config['Pitch']), yaw=float(camera_config['Yaw']))
            fov = float(camera_config['FOV'])

            calibration = np.identity(3)
            calibration[0, 2] = self.view_width / 2.0
            calibration[1, 2] = self.view_height / 2.0
            calibration[0, 0] = calibration[1, 1] = self.view_width / \
                (2.0 * np.tan(fov * np.pi / 360.0))
            
            camera_rgb = self.world.spawn_actor(
                self.get_camera_blueprint(fov, 'sensor.camera.rgb'),
                carla.Transform(location, rotation)
            )
            camera_rgb.calibration = calibration
            self.rgb_camera_list.append(('RGB_'+str(cam_id), camera_rgb))

            camera_semseg = self.world.spawn_actor(
                self.get_camera_blueprint(fov, 'sensor.camera.semantic_segmentation'),
                carla.Transform(location, rotation)
            )
            camera_semseg.calibration = calibration
            self.semseg_camera_list.append(('SS_' + str(cam_id), camera_semseg))
            self.camera_ids.append(cam_id)

    def get_camera_blueprint(self, fov, cam_type):
        camera_bp = self.world.get_blueprint_library().find(cam_type)
        camera_bp.set_attribute('image_size_x', str(self.view_width))
        camera_bp.set_attribute('image_size_y', str(self.view_height))
        camera_bp.set_attribute('fov', str(fov))
        camera_bp.set_attribute('sensor_tick', str(self.fixed_delta))
        return camera_bp

    def make_dirs(self, data_dir_path):
        self.data_dir_path = data_dir_path
        if not os.path.exists(data_dir_path):
            os.mkdir(data_dir_path)
        
        self.img_dir_path = os.path.join(data_dir_path, 'images')
        if not os.path.exists(self.img_dir_path):
            os.mkdir(self.img_dir_path)
        
        self.log_dir_path = os.path.join(data_dir_path, 'logs')
        if not os.path.exists(self.log_dir_path):
            os.mkdir(self.log_dir_path)

        self.result_dir_path = os.path.join(data_dir_path, 'output_tracks')
        if not os.path.exists(self.result_dir_path):
            os.mkdir(self.result_dir_path)
        

    def connect_client(self, port):
        self.port = port
        
        print('[INFO] Connecting to the client...')
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.client.load_world('Tracking_package')

        self.world = self.client.get_world()
        self._settings = self.world.get_settings()
        settings = self._settings
        settings.fixed_delta_seconds = self.fixed_delta
        settings.synchronous_mode = True
        settings.no_rendering = False
        self.world.apply_settings(settings)
        print('[INFO] Client Connection Established')

    def setup_camera_feed(self, track_id):
        print('[INFO] Setting up the camera feeds.')
        def make_queue(register_event, cam_id):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append((cam_id, q))

        make_queue(self.world.on_tick, 'system')
        for (rgb_cam_id, rgb_cam), (semseg_cam_id, semseg_cam) in zip(self.rgb_camera_list, self.semseg_camera_list):
            make_queue(rgb_cam.listen, rgb_cam_id)
            make_queue(semseg_cam.listen, semseg_cam_id)
            cam_num = int(rgb_cam_id.split('_')[1])
        print('[INFO] Camera feeds all set.')

    