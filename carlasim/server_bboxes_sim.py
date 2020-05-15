import glob
import sys

try:
    sys.path.append(
        glob.glob('carla-0.9.7-py3.5-linux-x86_64.egg')[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

# noinspection PyUnresolvedReferences
import carla

import logging
import time
import cv2
import csv
import queue
import random
import numpy as np
import argparse
import os
from collections import OrderedDict
import configparser
from bounding_boxes import ClientSideBoundingBoxes


start_locations = []
middle_locations = []
end_locations = []


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================
def xgen(start_end):
    return random.uniform(start_end[0], start_end[1])


def ygen(start_end):
    return random.uniform(start_end[0], start_end[1])


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None

        self.cameras = []
        self.semseg_cameras = []
        self.video_writers = []
        self.pedestrians = []
        self.paths = {}
        self.images = {}
        self.tracks = {}
        self.semseg_images = {}

        self.config = None
        self.args = None

        self.car = None
        self.image = None
        self.capture = True
        self.first_frame = 0
        self.cam_count = 2

        self.frames_count = {}

        self.view_width = 1280  # //2
        self.view_height = 720  # //2
        self.view_fov = 110
        self.fps = 15
        # self.G = map.Map().get_graph()
        # self.M = map.Map()

    def parse_config(self):
        argparser = argparse.ArgumentParser(
            description=__doc__)
        argparser.add_argument(
            '--host',
            metavar='H',
            default='localhost',
            help='IP of the host server (default: 143.248.55.230 - desktop PC)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '-n', '--number-of-vehicles',
            metavar='N',
            default=10,
            type=int,
            help='number of vehicles (default: 10)')
        argparser.add_argument(
            '-w', '--number-of-walkers',
            metavar='W',
            default=50,
            type=int,
            help='number of walkers (default: 50)')
        argparser.add_argument(
            '--safe',
            action='store_true',
            help='avoid spawning vehicles prone to accidents')
        argparser.add_argument(
            '--filterv',
            metavar='PATTERN',
            default='vehicle.*',
            help='vehicles filter (default: "vehicle.*")')
        argparser.add_argument(
            '--filterw',
            metavar='PATTERN',
            default='walker.pedestrian.*',
            help='pedestrians filter (default: "walker.pedestrian.*")')
        argparser.add_argument(
            '--config',
            default='10_camera_config.ini',
            help='path to the configuration file (default: "config.ini")')
        argparser.add_argument(
            '--foldername',
            default='videos',
            help='name of folder to save videos in')
        argparser.add_argument(
            '--likelihood',
            type=float,
            default=0.5,
            help='Likelihood of walker changing speed')

        self.args = argparser.parse_args()

        self.config = configparser.ConfigParser()
        self.config.read(self.args.config)
        if 'config.ini' not in self.args.config:
            fname = self.args.foldername
        else:
            fname = self.args.config.replace('_config.ini', '')
        self.cam_count = int(self.config['GENERAL']['Cameras'])

        self.view_width = int(self.config['GENERAL']['Image_Size_X'])
        self.view_height = int(self.config['GENERAL']['Image_Size_Y'])
        self.view_fov = int(self.config['GENERAL']['FOV'])
        self.fps = int(self.config['GENERAL']['FPS'])

    def connect_client(self, port=2000):
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(30.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1 / 15
        self.world.apply_settings(settings)

    def camera_blueprint(self, cam_type='sensor.camera.rgb'):
        """
        Returns camera blueprint.
        """
        # camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp = self.world.get_blueprint_library().find(cam_type)
        camera_bp.set_attribute('image_size_x', str(self.view_width))
        camera_bp.set_attribute('image_size_y', str(self.view_height))
        camera_bp.set_attribute('fov', str(self.view_fov))
        camera_bp.set_attribute('sensor_tick', str(1 / self.fps))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self, cam_id, location, writer, rotation, cam_type='rgb'):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        camera_transform = carla.Transform(location, rotation)
        if cam_type == 'rgb':
            camera = self.world.spawn_actor(self.camera_blueprint(
                cam_type='sensor.camera.rgb'), camera_transform)
            camera.listen(lambda image: self.get_rgb_image(
                cam_id, writer, image))
            self.cameras.append(camera)
        elif cam_type == 'semseg':
            camera = self.world.spawn_actor(self.camera_blueprint(cam_type='sensor.camera.semantic_segmentation'),
                                            camera_transform)
            camera.listen(lambda image: self.get_semseg_image(cam_id, image))
            self.semseg_cameras.append(camera)
        calibration = np.identity(3)
        calibration[0, 2] = self.view_width / 2.0
        calibration[1, 2] = self.view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.view_width / \
            (2.0 * np.tan(self.view_fov * np.pi / 360.0))
        self.cameras[cam_id].calibration = calibration

    def get_rgb_image(self, cam_id, writer, img):
        bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
            self.pedestrians, self.cameras[cam_id])
        image = ClientSideBoundingBoxes.process_img(
            img, self.view_width, self.view_height)
        # image = cv2.UMat(image)
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
                    point[0][0], point[0][1], width, height, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
                # * Crops box from the image and save image
                abs_left = point[0][0] - int(width / 2) - 5
                abs_top  = point[0][1] - int(height / 2) - 5
                # * Handle cases where box corners are out of image boundaries
                crop_left = max(0, abs_left)
                crop_top = max(0, abs_top)
                # * Adding + 10 as it takes the -5 from {abs_left} and {abs_top} into account 
                cropped = image[crop_top: min(self.view_height, int(abs_top + height + 10)), crop_left: min(self.view_width, int(abs_left + width + 10))]
                # TODO Should save this in a folder
                cv2.imwrite("cam_{}_frame{}.jpg".format(cam_id, img.frame), cropped)
        # * Not found
        else:
            self.tracks[cam_id][img.frame] = (-1, -1)
        self.frames_count[cam_id] += 1

    def get_semseg_image(self, cam_id, img):
        # ! Is there a better way to do this? Keep small stack/queue of last 5 semantic segmentation frames?
        # ! Currently, we are comparing frame X from semseg camera to frame X-1 from regular camera!!!
        # ? Use a stack based dictionary where we process the earliest frame (if possible) and whatever is left over is processed in finally
        self.semseg_images[cam_id].put((img.frame, img))

        index, image = self.semseg_images[cam_id].queue[0]

        try:
            pos = self.tracks[cam_id][index]
            self.semseg_images[cam_id].get()  # Pop out top item
            image = ClientSideBoundingBoxes.process_img(
                image, self.view_width, self.view_height)
            if pos != (-1, -1):
                try:
                    found = False
                    for i in range(-30, 30):
                        for j in range(-30, 30):
                            color = image[pos[1]+i, pos[0]+j]
                            if color[2] == 4:
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        self.tracks[cam_id][index] = (-1, -1)
                except IndexError:
                    self.tracks[cam_id][index] = (-1, -1)
                    pass
        except KeyError:
            pass

    def save_track(self, start_zone, end_zone, run):
        # TODO Get folder/filename as argument
        with open('novid/start_{}_end_{}_run_{}_track.csv'.format(start_zone, end_zone, run), mode='w') as file:
            writer = csv.writer(file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            hasTracks = True
            active_cams = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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

    def game_loop(self, path, start_zone, end_zone, run):
        """
        Main program loop.
        """

        try:
            global fname

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor
            failed = False
            walkers_list = []
            camera_list = []
            all_id = []

            # -------------
            # Spawn Cameras
            # -------------

            camera_transforms = []
            # TODO extract as function?
            for camera in range(self.cam_count):
                cam = self.config['CAMERA_' + str(camera + 1)]
                location = carla.Location(x=float(cam['PosX']), y=float(
                    cam['PosY']), z=float(cam['PosZ']))
                rotation = carla.Rotation(roll=float(cam['Roll']), pitch=float(
                    cam['Pitch']), yaw=float(cam['Yaw']))
                writer = None
                self.tracks[camera] = OrderedDict()
                self.semseg_images[camera] = queue.Queue()
                self.frames_count[camera] = 0
                self.setup_camera(camera, location, writer,
                                  rotation, cam_type='rgb')
                time.sleep(0.1)
                self.setup_camera(camera, location, writer,
                                  rotation, cam_type='semseg')
                time.sleep(0.1)
            # -------------
            # Spawn Walkers
            # -------------
            # 1. take all the random locations to spawn
            # TODO Need to add static people here
            spawn_points = []
            for i in range(1):
                spawn_point = carla.Transform()
                start_pos = path.pop(0)
                loc = carla.Location(x=start_pos[0], y=start_pos[1], z=1)
                start_locations.append(loc)
                if (loc != None):
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            # 2. we spawn the walker object
            batch = []
            cnt = 1
            for spawn_point in spawn_points:
                # Get blueprint from library; first because only one result is expected
                bp_name = self.config['PEDESTRIAN_' + str(cnt)]['Blueprint']
                walker_bp = self.world.get_blueprint_library().filter(bp_name)[
                    0]
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                batch.append(SpawnActor(walker_bp, spawn_point))
                cnt += 1
            results = self.client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                    failed = True
                else:
                    walkers_list.append({"id": results[i].actor_id})
            elapsed_frames = 0
            if not failed:
                # 3. we spawn the walker controller
                batch = []
                walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                for i in range(len(walkers_list)):
                    batch.append(SpawnActor(walker_controller_bp,
                                            carla.Transform(), walkers_list[i]["id"]))
                results = self.client.apply_batch_sync(batch, True)
                for i in range(len(results)):
                    if results[i].error:
                        # logging.error(results[i].error)
                        print(results[i].error)
                    else:
                        walkers_list[i]["con"] = results[i].actor_id
                # 4. we put altogether the walkers and controllers id to get the objects from their id
                for i in range(len(walkers_list)):
                    all_id.append(walkers_list[i]["con"])
                    all_id.append(walkers_list[i]["id"])
                all_actors = self.world.get_actors(all_id)
                cnt = 1
                # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
                for i in range(0, len(all_id), 2):
                    # start walker
                    all_actors[i].start()
                    # set walk to random point
                    pedestrian = self.config['PEDESTRIAN_' + str(cnt)]
                    # try Middle points
                    # * Convert to carla location type and assign to dictionary
                    path = [carla.Location(point[0], point[1], z=1)
                            for point in path]
                    # * Go to first waypoint
                    print("Going to {}".format(path[0]))
                    all_actors[i].go_to_location(path[0])
                    # * Get start & end position for pedestrian
                    # * Put into graph, get resulting path
                    # * Spawn at start point and then add rest of points to middle points list

                    # random max speed
                    if pedestrian['Speed'] != 'random':
                        all_actors[i].set_max_speed(float(pedestrian['Speed']))
                    else:
                        # max speed between 1 and 2 (default is 1.4 m/s)
                        all_actors[i].set_max_speed(random.uniform(1.2, 2))
                    cnt += 1


                self.pedestrians = self.world.get_actors().filter('walker.*')

                # video_writers = []

                stopgo = False
                start = time.time()
                sec_tick = 1
                self.world.wait_for_tick()

                while True:
                    cnt = 1
                    elapsed_frames += 1
                    for i in range(0, len(all_id), 2):
                        pedestrian = self.config['PEDESTRIAN_' + str(cnt)]
                        # only if middle point exists
                        current_location = all_actors[i].get_location()
                        if elapsed_frames > 300:
                            stopgo = True
                        if len(path) > 1:
                            if current_location.distance(path[0]) < 5:
                                print("Reached location {}".format(path.pop(0)))
                                print("Going to {}".format(path[0]))
                                all_actors[i].go_to_location(path[0])
                                print(len(path))
                                elapsed_frames = 0
                        else:
                            if current_location.distance(path[0]) < 5:
                                print("Reached end")
                                stopgo = True
                        cnt += 1
                    if stopgo == True:
                        break
                    self.world.wait_for_tick()

        finally:
            for camera in self.cameras:
                camera.destroy()
            for camera in self.semseg_cameras:
                camera.destroy()

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(all_id), 2):
                all_actors[i].stop()

            print('\ndestroying %d walkers' % len(walkers_list))
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in all_id])

            for cam in range(self.cam_count):
                while self.semseg_images[cam].empty() is False:
                    try:
                        index, image = self.semseg_images[cam].get()
                        pos = self.tracks[cam][index]
                        image = ClientSideBoundingBoxes.process_img(
                            image, self.view_width, self.view_height)
                        if pos != (-1, -1):
                            try:
                                color = image[pos[1], pos[0]]
                                if color[2] != 4:
                                    self.tracks[cam][index] = (-1, -1)
                            except IndexError:
                                pass
                    except KeyError:
                        print("Camera {} frame id mismatch".format(cam))

            if elapsed_frames < 300:
                self.save_track(start_zone, end_zone, run)  # TODO Uncomment

            for key, value in self.frames_count.items():
                print("Camera {} : Frames - {}".format(key, value))

            # TODO Print Ground Truth results

            time.sleep(0.5)


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
