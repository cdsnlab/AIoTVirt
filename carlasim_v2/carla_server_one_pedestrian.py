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

    def _retrieve_data(self, sensor_queue, timeout=2.0):
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                if data.frame == self.frame:
                    return data
            except queue.Empty:
                break

    def process_data_feeds(self, data_feeds):
        index = 1
        while index < len(data_feeds):
            rgb_cam_id, rgb_frame = data_feeds[index]
            semseg_cam_id, semseg_frame = data_feeds[index + 1]
            index += 2
            
            rgb_cam_type, rgb_cam_num = rgb_cam_id.split('_')
            rgb_cam_num = int(rgb_cam_num)
            semseg_cam_type, semseg_cam_num = semseg_cam_id.split('_')
            semseg_cam_num = int(semseg_cam_num)
            if (((rgb_cam_type == 'RGB') & (semseg_cam_type == 'SS')) & (rgb_cam_num == semseg_cam_num) & (rgb_frame.frame == semseg_frame.frame)):
                bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(self.pedestrians, self.rgb_camera_list[rgb_cam_num][1])

                # if rgb_cam_num == 9:
                #     rgb_image = ClientSideBoundingBoxes.process_img(rgb_frame, self.view_width, self.view_height)
                #     cv2.imwrite(os.path.join(self.track_img_dir, 'frame_{}_cam_{}_rgb.jpg'.format(rgb_frame.frame, rgb_cam_num)), rgb_image)

                #     semseg_frame.convert(carla.ColorConverter.CityScapesPalette)
                #     semseg_image = ClientSideBoundingBoxes.process_img(semseg_frame, self.view_width, self.view_height)
                #     cv2.imwrite(os.path.join(self.track_img_dir, 'frame_{}_cam_{}_semseg.jpg'.format(rgb_frame.frame, rgb_cam_num)), semseg_image)
                #print(bounding_boxes)
                if len(bounding_boxes) != 0:
                    box = bounding_boxes[0]
                    box = np.delete(box, 2, 1)
                    arr_box = np.asarray(box)
                    height = abs(arr_box[0][1] - arr_box[4][1])
                    width = abs(arr_box[0][0] - arr_box[3][0])
                    coords = [arr_box[3], arr_box[4]]
                    point = box.mean(0).getA().astype(int)
                    try:
                        # * Out of the camera's fov
                        if point[0][0] > self.view_width or point[0][1] > self.view_height or point[0][0] < 0 or point[0][1] < 0:
                            self.tracks[rgb_cam_num][rgb_frame.frame] = (-1, -1)

                        # inside the cam's fov
                        else:
                            # We first need to check if the semantic segmentation camera actually catches this walker
                            semseg_image = ClientSideBoundingBoxes.process_img(semseg_frame, self.view_width, self.view_height)
                            
                            found = False
                            color = semseg_image[point[0][1], point[0][0]]
                            if color[2] == 4:
                                found = True
                            

                            if found:
                                self.tracks[rgb_cam_num][rgb_frame.frame] = (rgb_frame.frame, point[0][0], point[0][1], width, height, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
                                rgb_image = ClientSideBoundingBoxes.process_img(rgb_frame, self.view_width, self.view_height)
                                if (self.record_mode == 1) | (self.record_mode == 3):
                                    cv2.imwrite(os.path.join(self.track_img_dir, 'frame_{}_cam_{}_rgb.jpg'.format(rgb_frame.frame, rgb_cam_num)), rgb_image)

                                if (self.record_mode == 2) | (self.record_mode == 3):
                                    semseg_frame.convert(carla.ColorConverter.CityScapesPalette)
                                    semseg_image = ClientSideBoundingBoxes.process_img(semseg_frame, self.view_width, self.view_height)
                                    cv2.imwrite(os.path.join(self.track_img_dir, 'frame_{}_cam_{}_semseg.jpg'.format(rgb_frame.frame, rgb_cam_num)), semseg_image)
                            else:
                                self.tracks[rgb_cam_num][rgb_frame.frame] = (-1, -1)
                    except IndexError:
                        pass
                else:
                    self.tracks[rgb_cam_num][rgb_frame.frame] = (-1, -1)

    def save_track(self):
        # TODO Get folder/filename as argument
        log_file_path = os.path.join(self.result_dir_path, self.track_id + '.csv')
        with open(log_file_path, mode='w') as file:
            writer = csv.writer(file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            hasTracks = True
            active_cams = self.camera_ids
            headers = ['Camera {}'.format(cam) for cam in active_cams]
            writer.writerow(['#'] + headers)
            # tracks = {cam: list(tr.values())
            #             for cam, tr in self.tracks.items()}
            for frame in self.tracks[-1]:
                try:
                    row = [self.tracks[cam][int(frame)] for cam in active_cams]
                    writer.writerow([int(frame)] + row)
                except KeyError:
                    pass
        
                # try:
                #     row = [tracks[cam][rowNumber]
                #             for cam in range(self.cam_count)]
                #     writer.writerow([rowNumber] + row)
                # except IndexError:
                #     pass
        with open('data/run_logs/{}.txt'.format(self.scrip_name), 'a') as log_file:
            log_file.write(self.track_id + '\n')
        log_file.close()


    def game_loop(self, track_id, track):

        print('[INFO] Start track simulation. ID: {}'.format(track_id))
        
        try:
            for cam_id in range(self.cam_count):
                self.tracks[cam_id] = OrderedDict()
            self.tracks[-1] = []

            if self.num_of_tracks_done == 0:
                self.setup_camera_feed(track_id)

            # Get the track id and create a folder to contain the output data
            self.track_id = track_id
            self.track_img_dir = os.path.join(self.img_dir_path, self.track_id)

            if not os.path.exists(self.track_img_dir):
                os.mkdir(self.track_img_dir)
                print('[INFO] [{}] Track\'s image folder created.'.format(track_id))

            self.track_done = False
            all_walkers = []

            #clock = pygame.time.Clock()

            num_of_points_passed = -1
            elapsed_frame = 0
            should_stop = False
            recorded_frame = 0
            while True:
                #clock.tick()
                
                self.frame = self.world.tick()
                recorded_frame += 1

                # During the first handful of frames, the feeds are not yet stablized, so we give them some time.
                if recorded_frame > 14:
                    # If there is currently no points passed, taht means we need to spawn the walker
                    # The first point to pass will be the spawn point
                    if num_of_points_passed == -1:
                        # Designating the spawn point as the first point in the path
                        spawn_point = carla.Transform()
                        spawn_coor = track.pop(0)
                        spawn_point.location = carla.Location(spawn_coor[0], spawn_coor[1], 1)


                        # Getting the blueprint for the walker
                        walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
                        walker_bp = random.choice(walker_blueprints)
                        walker_bp_id = walker_bp.id
    

                        if walker_bp.has_attribute('is_invincible'):
                            walker_bp.set_attribute('is_invincible', 'false')
                        
                        # Spawning the pedestrian model
                        pedes_model = self.world.try_spawn_actor(walker_bp, spawn_point)

                        self.frame = self.world.tick()

                        failed = True
                        if pedes_model:
                            all_walkers.append(pedes_model)
                            print('[INFO] [{}] Pedestrian Model Spawned at ({}, {}) - using blueprint {}'.format(track_id, spawn_coor[0], spawn_coor[1], walker_bp_id))
                            failed = False
                        else:
                            print('[INFO] [{}] Failed to spawn the pedestrian model.'.format(track_id))

                        if not failed:
                            # Spawning the AI controller walker, which is invisible
                            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                            ai_walker_model = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), pedes_model)

                            # Adding the pedestrian and ai walker models to the list of all ids so they can be deleted easily later
                            all_walkers.append(ai_walker_model)

                            # Start the ai walker controller model
                            ai_walker_model.start()
                            self.pedestrians = self.world.get_actors().filter(walker_bp_id)
                            walker_height = pedes_model.get_location().z

                            num_of_points_passed += 1

                            next_location = carla.Location(track[0][0], track[0][1], walker_height)
                            ai_walker_model.go_to_location(next_location)
                            ai_walker_model.set_max_speed(track[0][2])

                            print('[INFO] [{}] Walking to {}'.format(track_id, next_location))
                            num_of_points_passed += 1
                    elif num_of_points_passed == 0:
                        True
                    else:
                        elapsed_frame += 1
                        #clock.tick()

                        ai_walker_model.go_to_location(next_location)
                        current_location = ai_walker_model.get_location()
                        #print(current_location, current_location.distance(next_location))
                        if elapsed_frame > 100:
                            if len(track) > 1:
                                track.pop(0)
                                
                                next_location = carla.Location(track[0][0], track[0][1], walker_height)
                                ai_walker_model.go_to_location(next_location)
                                ai_walker_model.set_max_speed(track[0][2])

                                print("[INFO] [{}] Going to {}, {} remaining points - Current location {}".format(track_id, next_location, len(track), current_location))

                                elapsed_frame = 0
                            else: 
                                print("[INFO] [{}] Reached the end of the path!".format(track_id))
                                self.track_done = True
                                should_stop = True
                        if len(track) > 1:
                            if current_location.distance(next_location) < 3:
                                print("[INFO] [{}] Reached location {}".format(track_id, next_location))
                                track.pop(0)

                                next_location = carla.Location(track[0][0], track[0][1], walker_height)
                                ai_walker_model.go_to_location(next_location)
                                ai_walker_model.set_max_speed(track[0][2])

                                print("[INFO] [{}] Going to {}, {} remaining points - Current location {}".format(track_id, next_location, len(track), current_location))
                                elapsed_frame = 0
                        else:
                            if current_location.distance(next_location) < 3:
                                print("[INFO] [{}] Reached the end of the path!".format(track_id))
                                self.track_done = True
                                should_stop = True
                    if recorded_frame % 1 == 0:
                        try:
                            data_feeds = [(cam_id, self._retrieve_data(q)) for cam_id, q in self._queues]
                            #print(data)
                            assert all(x.frame == self.frame for _, x in data_feeds)
                            self.process_data_feeds(data_feeds)
                            self.tracks[-1].append(self.frame)
                        except AttributeError:
                            should_stop = True
                if should_stop:
                    break

        finally:
            ai_walker_model.stop()
            for walker in all_walkers:
                walker.destroy()
            self.world.tick()
            if self.track_done:
                self.save_track()
            self.num_of_tracks_done += 1
            time.sleep(2.0)
            
    
        #failed = False

            