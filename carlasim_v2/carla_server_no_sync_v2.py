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

class BasicSynchronousClient(object):
   """
   Basic implementation of a synchronous client.
   """

   def __init__(self, data_dir):
      #os.mkdir("data/start_{0}_end_{1}_run_{2}".format(self.start_zone, self.end_zone, self.run))
      self.client = None
      self.world = None

      self.cameras = []
      self.cameras_id = []
      self.semseg_cameras = []
      self.video_writers = []
      self.pedestrians = []
      self.tracks = {}
      self.rgb_images = {}
      self.semseg_images = {}
      self.semseg_status = {}

      self.config = None

      self.capture = True
      self.first_frame = 0

      self.frames_count = {}
      
      self.speed = 0
      self.data_dir = data_dir
      self.image_data_dir = os.path.join(self.data_dir, 'images')
      self.log_data_dir = os.path.join(self.data_dir, 'logs')

      self.image_map = {cam: set() for cam in range(10)}

   def parse_config(self, config_file):
      print('[INFO] Parsing the configurations...')
      self.config = configparser.ConfigParser()
      self.config.read(config_file)
      
      self.cam_count = int(self.config['GENERAL']['Cameras'])

      self.view_width = int(self.config['GENERAL']['Image_Size_X'])
      self.view_height = int(self.config['GENERAL']['Image_Size_Y'])
      self.view_fov = int(self.config['GENERAL']['FOV'])
      self.fps = int(self.config['GENERAL']['FPS'])
      print('[INFO] Configurations Parsed')


   def connect_client(self, port):
      print('[INFO] Connecting to the client...')
      self.client = carla.Client('localhost', port)
      self.client.set_timeout(30.0)
      self.world = self.client.get_world()
      print(self.world)
      settings = self.world.get_settings()
      settings.fixed_delta_seconds = 0.05
      #settings.synchronous_mode = True
      self.world.apply_settings(settings)
      print('[INFO] Client Connection Established')

   def install_camera(self):
      print('[INFO] Installing the cameras...')
      for camera in range(self.cam_count):
         self.cameras_id.append(camera)
         cam = self.config['CAMERA_' + str(camera + 1)]
         location = carla.Location(x=float(cam['PosX']), y=float(cam['PosY']), z=float(cam['PosZ']))
         rotation = carla.Rotation(roll=float(cam['Roll']), pitch=float(cam['Pitch']), yaw=float(cam['Yaw']))
         fov = float(cam['FOV'])
         writer = None
         self.tracks[camera] = OrderedDict()
         self.semseg_images[camera] = queue.Queue()
         self.frames_count[camera] = 0
         self.setup_camera(camera, location, writer, rotation, fov, cam_type='semseg')
         time.sleep(0.1)
         self.setup_camera(camera, location, writer, rotation, fov, cam_type='rgb')
         time.sleep(0.1)
      print('[INFO] Cameras Installation Done')

   def destroy_camera(self):
      print('[INFO] Destroying the cameras...')
      for camera in self.cameras:
            camera.destroy()
      for camera in self.semseg_cameras:
            camera.destroy()
      print('[INFO] Camera Detroyed')
   

   def set_synchronous_mode(self, synchronous_mode):
      """
      Sets synchronous mode.
      """

      settings = self.world.get_settings()
      settings.synchronous_mode = synchronous_mode
      self.world.apply_settings(settings)
      
   def get_rgb_image(self, cam_id, writer, img):
      

      bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
         self.pedestrians, self.cameras[cam_id])
      image = ClientSideBoundingBoxes.process_img(
         img, self.view_width, self.view_height)
      #print('rgb {}'.format(img.frame))
      # image = cv2.UMat(image)

      #cv2.imwrite(os.path.join('images', 'cam_{}_frame_{}.jpg'.format(cam_id, img.frame)), image)
      #cv2.imwrite("images/" + str(curr_time) + '.jpg', image)
      if len(bounding_boxes) != 0:
         box = bounding_boxes[0]
         box = np.delete(box, 2, 1)
         arr_box = np.asarray(box)
         height = abs(arr_box[0][1] - arr_box[4][1])
         width = abs(arr_box[0][0] - arr_box[3][0])
         coords = [arr_box[3], arr_box[4]]
         point = box.mean(0).getA().astype(int)
         # * Out of image boundaries
         if point[0][0] > self.view_width or point[0][1] > self.view_height or point[0][0] < 0 or point[0][1] < 0:
               self.tracks[cam_id][img.frame] = (-1, -1)
         # * Visible and in-image
         else:
               self.tracks[cam_id][img.frame] = (
                  img.frame, point[0][0], point[0][1], width, height, coords[0][0], coords[0][1], coords[1][0], coords[1][1], self.speed)
               # * Crops box from the image and save image
               abs_left = point[0][0] - int(width / 2) - 5
               abs_top  = point[0][1] - int(height / 2) - 5
               # * Handle cases where box corners are out of image boundaries
               crop_left = max(0, abs_left)
               crop_top = max(0, abs_top)
               # * Adding + 10 as it takes the -5 from {abs_left} and {abs_top} into account 
               cropped = image[crop_top: min(self.view_height, int(abs_top + height + 10)), crop_left: min(self.view_width, int(abs_left + width + 10))]
               # TODO Should save this in a folder
               # cv2.imwrite("data/start_{}_end_{}_run_{}/cam_{}_frame_{}.jpg".format(cam_id, img.frame), cropped)
               cv2.imwrite(os.path.join(self.track_img_dir, 'cam_{}_frame_{}.jpg'.format(cam_id, img.frame)), image)
      # * Not found
      else:
         self.tracks[cam_id][img.frame] = (-1, -1)

   def get_semseg_image(self, cam_id, img):
      # ! Is there a better way to do this? Keep small stack/queue of last 5 semantic segmentation frames?
      # ! Currently, we are comparing frame X from semseg camera to frame X-1 from regular camera!!!
      # ? Use a stack based dictionary where we process the earliest frame (if possible) and whatever is left over is processed in finally
      
      #img.save_to_disk(os.path.join('images', 'cam_{}_frame_{}.jpg'.format(cam_id, img.frame)), color_converter = carla.ColorConverter.CityScapesPalette)
      
      self.semseg_images[cam_id].put((img.frame, img))
      index, image = self.semseg_images[cam_id].queue[0]
      #temp, _ = self.semseg_images[cam_id].get()
      #print(img.frame, index)
      #index = img.frame

      #print(img.frame, index)
      print(img.frame)
      try:
         print(self.tracks[cam_id][index])
         pos = self.tracks[cam_id][index]
         #print('done', pos[0])
         #self.semseg_images[cam_id].get()  # Pop out top item
         image = ClientSideBoundingBoxes.process_img(
               img, self.view_width, self.view_height)
         #cv2.imwrite(os.path.join('images', 'cam_{}_frame_{}.jpg'.format(cam_id, img.frame)), image)
         #print(pos)
         if pos != (-1, -1):
            try:
               found = False
               for i in range(-10, 10):
                  for j in range(-10, 10):
                        color = image[pos[2]+i, pos[1]+j]
                        if color[2] == 4:
                           found = True
                           self.image_map[cam_id].remove(index)
                           break
                  if found:
                        break
               if not found:
                  self.tracks[cam_id][index] = (-1, -1)
            except IndexError:
               self.tracks[cam_id][index] = (-1, -1)
               pass
      except KeyError:
         #print('key error')
         #print('in', img.frame)
         pass
   
   def save_track(self):
      # TODO Get folder/filename as argument
      log_file_path = os.path.join(self.log_data_dir, self.track_id + '.csv')
      with open(log_file_path, mode='w') as file:
         writer = csv.writer(file, delimiter=',',
                              quotechar='"', quoting=csv.QUOTE_MINIMAL)
         hasTracks = True
         active_cams = self.cameras_id
         headers = ['Camera {}'.format(cam) for cam in active_cams]
         writer.writerow(['Frame'] + headers)
         tracks = {cam: list(tr.values())
                     for cam, tr in self.tracks.items()}
         for rowNumber in range(len(self.tracks[0])):
            try:
               row = [tracks[cam][rowNumber]
                     for cam in range(self.cam_count)]
               writer.writerow([rowNumber] + row)
            except IndexError:
               pass

   def remove_blank_images(self):
      """ At the end of the simulation, we are deleting the images which have been found to not
      contain the tracked target (by the semseg camera)
      """
      for camera, frames in self.image_map.items():
         for frame in frames:
               try:
                  img_path = os.path.join(self.track_img_dir, 'cam_{}_frame_{}.jpg'.format(camera, frame))
                  os.remove(img_path)
               except:
                  pass
      pass

   def setup_camera(self, cam_id, location, writer, rotation, fov, cam_type='rgb'):
      """
      Spawns actor-camera to be used to render view.
      Sets calibration for client-side boxes rendering.
      """
      camera_transform = carla.Transform(location, rotation)
      if cam_type == 'rgb':
         camera = self.world.spawn_actor(self.camera_blueprint(fov, 
               cam_type='sensor.camera.rgb'), camera_transform)
         camera.listen(lambda image: self.get_rgb_image(cam_id, writer, image))
         self.cameras.append(camera)
      elif cam_type == 'semseg':
         camera = self.world.spawn_actor(self.camera_blueprint(fov, cam_type='sensor.camera.semantic_segmentation'),
                                          camera_transform)
         camera.listen(lambda image: self.get_semseg_image(cam_id, image))
         self.semseg_cameras.append(camera)
      calibration = np.identity(3)
      calibration[0, 2] = self.view_width / 2.0
      calibration[1, 2] = self.view_height / 2.0
      calibration[0, 0] = calibration[1, 1] = self.view_width / \
         (2.0 * np.tan(fov * np.pi / 360.0))
      self.cameras[cam_id].calibration = calibration

   def camera_blueprint(self, fov, cam_type='sensor.camera.rgb'):
      """
      Returns camera blueprint.
      """
      # camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
      camera_bp = self.world.get_blueprint_library().find(cam_type)
      camera_bp.set_attribute('image_size_x', str(self.view_width))
      camera_bp.set_attribute('image_size_y', str(self.view_height))
      camera_bp.set_attribute('fov', str(fov))
      camera_bp.set_attribute('sensor_tick', str(1 / self.fps))
      return camera_bp

   def game_loop(self, track_id, path):

      for camera in range(self.cam_count):
         self.cameras_id.append(camera)
         cam = self.config['CAMERA_' + str(camera + 1)]
         location = carla.Location(x=float(cam['PosX']), y=float(
            cam['PosY']), z=float(cam['PosZ']))
         rotation = carla.Rotation(roll=float(cam['Roll']), pitch=float(
            cam['Pitch']), yaw=float(cam['Yaw']))
         fov = float(cam['FOV'])
         writer = None
         self.tracks[camera] = OrderedDict()
         self.semseg_images[camera] = queue.Queue(0)
         self.semseg_status[camera] = OrderedDict()
         self.frames_count[camera] = 0
         self.setup_camera(camera, location, writer, rotation, fov, cam_type='rgb')
         time.sleep(0.1)
         self.setup_camera(camera, location, writer, rotation, fov, cam_type='semseg')
         time.sleep(0.1)
         

      dummy_track = [(-73.5, 61.8, 1), (-73.5, 85.8, 1)]#[(-68.2, -3.1, 1), (-51.2, -3.6, 1)]#[(-68.2, -3.1, 1), (-51.2, -3.6, 1)]#[(-90.3,61.5, 1), (-87.6, 41.1, 1), (-90.3, 27.2, 1)]
      path = dummy_track

      self.track_id = track_id

      self.track_id = 'dummy'

      self.track_img_dir = os.path.join(self.image_data_dir, self.track_id)
      if not os.path.exists(self.track_img_dir):
         os.mkdir(self.track_img_dir)

      #self.image_map = {cam: set() for cam in range(10)}
      #for cam_id in self.cameras_id:
      #   self.tracks[cam_id] = OrderedDict()
      #self.semseg_images[cam_id] = queue.Queue()

      # Initiating the walker list and id list
      walkers_list = []
      all_id = []

      failed = False

      # Converting the path to carla coordinates
      #path = [carla.Location(point[0], point[1], 1) for point in path]

      # Designating the spawn point as the first point in the path
      spawn_point = carla.Transform()
      spawn_coor = path.pop(0)
      print(spawn_coor)
      spawn_location = carla.Location(spawn_coor[0], spawn_coor[1], 1)
      print(spawn_location)
      spawn_point.location = spawn_location

      # Getting the blueprint for the walker
      walker_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")
      walker_bp = random.choice(walker_blueprints)
      walker_bp_id = walker_bp.id
      print(walker_bp_id)

      #bp_name = self.config['PEDESTRIAN_1']['Blueprint']
      #walker_bp = self.world.get_blueprint_library().filter(bp_name)[0]

      
      # If the walker is invincible, set that to false
      if walker_bp.has_attribute('is_invincible'):
         walker_bp.set_attribute('is_invincible', 'false')

      # Spawning the model
      pedes_model = self.world.try_spawn_actor(walker_bp, spawn_point)
      if pedes_model:
         walkers_list.append({"id": pedes_model.id})
         all_id.append(pedes_model)
         print('[INFO] Spawning Pedestrian model Succeeded')
         failed = False
      else:
         print('[ERROR] Spawning Pedestrian model Failed')
         failed = True
      
      if not failed:
         # Spawning the AI controller walker, which is invisible
         walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
         ai_walker_model = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), pedes_model)

         # Adding the pedestrian and ai walker models to the list of all ids so they can be deleted easily later
         all_id.append(ai_walker_model)

         # Start the ai model
         ai_walker_model.start()
         cnt = 1

         stopgo = False
         start = time.time()
         sec_tick = 1
         self.world.wait_for_tick()

         self.pedestrians = self.world.get_actors().filter(walker_bp_id)
         elapsed_frames = 0

         next_location = carla.Location(path[0][0], path[0][1], 1)
         ai_walker_model.go_to_location(next_location)
         ai_walker_model.set_max_speed(path[0][2])
         print("Going to {}".format(next_location))

         while not stopgo:
            elapsed_frames += 1

            current_location = ai_walker_model.get_location()
            if elapsed_frames > 500:
               stopgo = True
            if len(path) > 1:
               if current_location.distance(next_location) < 1:
                  print(current_location.distance(next_location))
                  print("Reached location {}".format(path.pop(0)))

                  next_location = carla.Location(path[0][0], path[0][1], 1)
                  ai_walker_model.go_to_location(next_location)
                  ai_walker_model.set_max_speed(path[0][2])
                  print("Going to {}, {} remaining points".format(next_location, len(path)))
                  elapsed_frames = 0
            else:
               if current_location.distance(next_location) < 5:
                  print("Reached the end of the path!")
                  stopgo = True
            self.world.wait_for_tick()
         #print(ai_walker_model.get_location())
      
      self.destroy_camera()

      ai_walker_model.stop()
      destroyer = []
      for x in all_id:
         #print(x)
         destroyer.append(carla.command.DestroyActor(x))
      self.client.apply_batch(destroyer)

      for cam in range(self.cam_count):
         print(self.semseg_images[cam].qsize())
         while self.semseg_images[cam].empty() is False:
            try:
               index, image = self.semseg_images[cam].get()
               pos = self.tracks[cam][index]
               image = ClientSideBoundingBoxes.process_img(
                     image, self.view_width, self.view_height)

               print('last', pos[0])
               if pos != (-1, -1):
                  try:
                     color = image[pos[2], pos[1]]
                     if color[2] != 4:
                        self.tracks[cam][index] = (-1, -1)
                        self.semseg_status[cam][index] = 'not found'
                     else:
                        self.image_map[cam].remove(index)
                        self.semseg_status[cam][index] = 'found'
                  except IndexError:
                     pass
            except KeyError:
               #print("Camera {} frame id mismatch".format(cam))
               pass
         
         #print(self.semseg_status[cam])
         # for frame_num in self.tracks[cam]:
         #    try:
         #       if self.semseg_status[cam][frame_num] == 'not found':
         #          self.tracks[cam][frame_num] = (-1, -1)
         #    except KeyError:

         #       self.tracks[cam][frame_num] = (-1, -1)
      
      
      self.save_track()
      self.remove_blank_images()
      

      time.sleep(3)
         


