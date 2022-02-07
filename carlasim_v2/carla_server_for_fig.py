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
from pedestrian import Pedestrian

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
        self.pedestrian_list = []
        self.camera_ids = []
        self.num_of_tracks_done = 0
        self.record_mode = 0#record_mode
        self.scrip_name = script_name
        self.run_log_path = os.path.join(data_dir_path, 'run_logs', script_name + '.txt')

        self.parse_cam_config(config_file_path)
        self.make_dirs(data_dir_path)
        self.connect_client(port)
        self.setup_camera()
    
        #self.image_map = {cam: set() for cam in range()}

    def parse_cam_config(self, config_file_path):
        print('[INFO] Parsing the configurations...')
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)
        
        self.num_cams = int(self.config['GENERAL']['Cameras'])

        self.view_width = int(self.config['GENERAL']['Image_Size_X'])
        self.view_height = int(self.config['GENERAL']['Image_Size_Y'])
        self.view_fov = int(self.config['GENERAL']['FOV'])
        self.fps = int(self.config['GENERAL']['FPS'])
        self.fixed_delta = 1 / self.fps


        self.rgb_camera_list = []
        self.semseg_camera_list = []
        print('[INFO] Configurations Parsed')

    def setup_camera(self):
        for cam_id in range(self.num_cams):
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
            if rgb_cam_num not in [0, 1, 4]:
                continue
            semseg_cam_type, semseg_cam_num = semseg_cam_id.split('_')
            semseg_cam_num = int(semseg_cam_num)
            if (((rgb_cam_type == 'RGB') & (semseg_cam_type == 'SS')) & (rgb_cam_num == semseg_cam_num) & (rgb_frame.frame == semseg_frame.frame)):
                for pedestrian in self.pedestrian_list:
                    if str(pedestrian.id) != '1':
                        continue
                    bounding_box = ClientSideBoundingBoxes.get_bounding_box(pedestrian.carla_actor, self.rgb_camera_list[rgb_cam_num][1])
                    
                    # if int(rgb_cam_num) in [0]:
                    #     rgb_image = ClientSideBoundingBoxes.process_img(rgb_frame, self.view_width, self.view_height)
                    #     cv2.imwrite(os.path.join(self.img_dir_path, 'frame_{}_cam_{}_rgb.jpg'.format(rgb_frame.frame, rgb_cam_num)), rgb_image)


                    #     semseg_frame.convert(carla.ColorConverter.CityScapesPalette)
                    #     semseg_image = ClientSideBoundingBoxes.process_img(semseg_frame, self.view_width, self.view_height)
                    #     cv2.imwrite(os.path.join(self.img_dir_path, 'frame_{}_cam_{}_semseg.jpg'.format(rgb_frame.frame, rgb_cam_num)), semseg_image)
                    #print(bounding_boxes)
                    if bounding_box is not None:
                        print(bounding_box)
                        bounding_box = np.delete(bounding_box, 2, 1)
                        print(self.frame, rgb_cam_num)
                        print(bounding_box)
                        arr_box = np.asarray(bounding_box)
                        height = abs(arr_box[0][1] - arr_box[4][1])
                        width = abs(arr_box[0][0] - arr_box[3][0])
                        max_x = -1
                        min_x = 9999
                        max_y = -1
                        min_y = 9999
                        for p in arr_box:
                            max_x = max(p[0], max_x)
                            min_x = min(p[0], min_x)
                            max_y = max(p[1], max_y)
                            min_y = min(p[1], min_y)

                        coords = ((min_x, min_y), (max_x, max_y))
                        point = bounding_box.mean(0).getA().astype(int)


                        try:
                            # * Out of the camera's fov
                            if (point[0][0] > self.view_width) or (point[0][1] > self.view_height) or (point[0][0] < 0) or (point[0][1] < 0):
                                pedestrian.recorded_track[rgb_cam_num].append((-1, -1))

                            # Inside the cam fov
                            else:
                                # We first need to check if the semantic segmentation camera actually catches this walker
                                semseg_image = ClientSideBoundingBoxes.process_img(semseg_frame, self.view_width, self.view_height)
                                found = False

                                color = semseg_image[point[0][1], point[0][0]]
                                if color[2] == 4:
                                    found = True
                                if found:
                                    rgb_image = ClientSideBoundingBoxes.process_img(rgb_frame, self.view_width, self.view_height)
                                    rgb_image = ClientSideBoundingBoxes.draw_bounding_box(rgb_image, coords, width, height)
                                    cv2.imwrite(os.path.join(self.img_dir_path, 'frame_{}_cam_{}_rgb_bb.jpg'.format(rgb_frame.frame + 935, rgb_cam_num)), rgb_image)
                                    
                                    pedestrian.recorded_track[rgb_cam_num].append((
                                        rgb_frame.frame, 
                                        point[0][0], 
                                        point[0][1], 
                                        width, 
                                        height, 
                                        coords[0][0], 
                                        coords[0][1], 
                                        coords[1][0], 
                                        coords[1][1]
                                    ))
                                    if (self.record_mode == 1) | (self.record_mode == 3):
                                        rgb_image = ClientSideBoundingBoxes.process_img(rgb_frame, self.view_width, self.view_height)
                                        rgb_image = ClientSideBoundingBoxes.draw_bounding_box(rgb_image, coords, width, height)
                                        cv2.imwrite(os.path.join(self.img_dir_path, 'frame_{}_cam_{}_rgb.jpg'.format(rgb_frame.frame, rgb_cam_num)), rgb_image)

                                    if (self.record_mode == 2) | (self.record_mode == 3):
                                        rgb_image = ClientSideBoundingBoxes.process_img(rgb_frame, self.view_width, self.view_height)
                                        semseg_frame.convert(carla.ColorConverter.CityScapesPalette)
                                        semseg_image = ClientSideBoundingBoxes.process_img(semseg_frame, self.view_width, self.view_height)
                                        cv2.imwrite(os.path.join(self.img_dir_path, 'frame_{}_cam_{}_semseg.jpg'.format(rgb_frame.frame, rgb_cam_num)), semseg_image)
                                    
                                else:
                                    pedestrian.recorded_track[rgb_cam_num].append((-1, -1))
                        except:
                            pedestrian.recorded_track[rgb_cam_num].append((-1, -1))
                            pass
                            #self.logging('error ' + str(rgb_cam_num.frame), pedestrian.id)
                    else:
                        pedestrian.recorded_track[rgb_cam_num].append((-1, -1))
                        #self.logging('no bb', pedestrian.id)
    def logging(self, msg, pedestrian_id = None):
        print(msg)


    def setup_camera_feed(self):
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


    def game_loop(self, track_list, CONCURRENT_TRACKS = 20):
        try:

            num_completed_tracks = 0 #number of completed tracks
            num_tracks = len(track_list) #total number of tracks to be simulated
            index = 0 # The index that are going to be added to the system next

            # Setup the cameras
            self.setup_camera_feed()
            self.frame = 0

            while self.frame < 49:
                self.frame = self.world.tick()

            # Since objects in carla have to wait for a world tick to come to life, we use a dictionary list to store pedestrian attributes
            # temporarily before the next tick
            self.pedes_wait_list = []

            print('[INFO] Start simulation.')

            # If all the tracks are not simulated
            should_stop = False
            while True: #(len(track_list) > 0) or (len(self.pedestrian_list) > 0): 
                self.frame = self.world.tick()
                # If the current number of  tracks in the system has not reached the maximum
                while len(self.pedes_wait_list) > 0:
                    waiting_pedestrian = self.pedes_wait_list.pop(0)
                    walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    ai_walker_model = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), waiting_pedestrian['actor'])
                    pedes = Pedestrian(
                        track_id = waiting_pedestrian['id'],
                        walking_track = waiting_pedestrian['track'],
                        height = waiting_pedestrian['actor'].get_location().z,
                        carla_id = waiting_pedestrian['actor'].id,
                        carla_actor = waiting_pedestrian['actor'],
                        ai_controller = ai_walker_model,
                        img_dir_path = self.img_dir_path,
                        result_dir_path = self.result_dir_path,
                        log_dir_path = self.log_dir_path,
                        num_cams = self.num_cams,
                        first_appear_frame = self.frame
                    )
                    log_msg = '[SIM] Adding a pedestrian to the PEDESTRIAN LIST. Track ID: {}. Carla ID: {}'.format(
                        waiting_pedestrian['id'], 
                        waiting_pedestrian['actor'].id
                    )
                    pedes.logging(log_msg)
                    self.logging(log_msg, pedes.id)
                    self.pedestrian_list.append(pedes)


                while (len(track_list) > 0) and (len(self.pedestrian_list) + len(self.pedes_wait_list) < CONCURRENT_TRACKS):
                   
                        # Add a pedestrian to the syste
                        new_track = track_list.pop(0)

                        new_pedestrian = {}
                        new_pedestrian['track'] = new_track[1]
                        new_pedestrian['id'] = new_track[0]
                        index += 1

                        # Get a new random blue print
                        if new_pedestrian['id'] == '1':
                            print(new_track[1])
                            new_blueprint = random.choice(
                                self.world.get_blueprint_library().filter('walker.pedestrian.0005')
                            )    
                        else:
                            new_blueprint = random.choice(
                                self.world.get_blueprint_library().filter('walker.pedestrian.*')
                            )
                        print(new_track[0], new_blueprint)
                        if new_blueprint.has_attribute('is_invincible'):
                            new_blueprint.set_attribute('is_invincible', 'false')

                        # Spawning an actor 
                        spawn_coor = new_pedestrian['track'].pop(0)
                        print(spawn_coor)
                        new_spawn_point = carla.Transform()
                        new_spawn_point.location = carla.Location(spawn_coor[0], spawn_coor[1], 1)

                        new_pedestrian['actor'] = self.world.try_spawn_actor(new_blueprint, new_spawn_point)
                        
                        if new_pedestrian['actor']:
                            self.pedes_wait_list.append(new_pedestrian)
                            self.logging(
                                '[SIM] Adding a pedestrian to the WAIT LIST. Track ID: {}.'.format(new_pedestrian['id']), 
                                pedestrian_id = new_pedestrian['id']
                            )

                    #new_pedestrian = pedestrian()
                
                for pedestrian in self.pedestrian_list:
                    pedestrian.elapsed_frame += 1
                    pedestrian.walk()

                    if pedestrian.elapsed_frame > 300:
                        if len(pedestrian.walking_track) > 1:
                            pedestrian.get_next_location()
                            pedestrian.walk()

                            log_msg = '[SIM] [{}] Going to {}, {} points remain - Current location {}'.format(
                                pedestrian.id, 
                                pedestrian.next_location,
                                len(pedestrian.walking_track),
                                pedestrian.carla_actor.get_location())
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)

                            pedestrian.elapsed_frame = 0
                        else:
                            log_msg = '[SIM] [{}] Reached the end of the path!'.format(pedestrian.id)
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)
                            pedestrian.complete_track()
                            self.pedestrian_list.remove(pedestrian)

                            num_completed_tracks += 1
                            with open(self.run_log_path, 'a') as run_log_file:
                                run_log_file.write('{}\n'.format(pedestrian.id))
                                run_log_file.close()

                            log_msg = '[SIM] [{}] Removed from pedestrian list - {} remain!'.format(pedestrian.id, num_tracks - num_completed_tracks)
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)
                            # Done track

                    if len(pedestrian.walking_track) > 1:
                        if pedestrian.arrive_at_next_location():
                            log_msg = '[SIM] [{}] Reached {}!'.format(pedestrian.id, pedestrian.next_location)
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)
                            pedestrian.get_next_location()
                            pedestrian.walk()

                            log_msg = '[SIM] [{}] Going to {}, {} points remain - Current location {}'.format(
                                pedestrian.id, 
                                pedestrian.next_location,
                                len(pedestrian.walking_track),
                                pedestrian.carla_actor.get_location())
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)

                            pedestrian.elapsed_frame = 0
                    else:
                        if pedestrian.arrive_at_next_location():
                            log_msg = '[SIM] [{}] Reached the end of the path!'.format(pedestrian.id)
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)
                            pedestrian.complete_track()
                            self.pedestrian_list.remove(pedestrian)

                            num_completed_tracks += 1
                            with open(self.run_log_path, 'a') as run_log_file:
                                run_log_file.write('{}\n'.format(pedestrian.id))
                                run_log_file.close()

                            log_msg = '[SIM] [{}] Removed from pedestrian list - {} remain!'.format(pedestrian.id, num_tracks - num_completed_tracks)
                            self.logging(log_msg, pedestrian.id)
                            pedestrian.logging(log_msg)

                if self.frame % 1 == 0:
                    try:
                        data_feeds = [(cam_id, self._retrieve_data(q)) for cam_id, q in self._queues]

                        assert all(x.frame == self.frame for _, x in data_feeds)
                        self.process_data_feeds(data_feeds)

                    except AttributeError:
                        should_stop = True
                        print('[INFO] Attribute Error happened when accessing camera feeds')
                if should_stop:
                    break


            
        finally:
            True

if __name__ == '__main__':
    True