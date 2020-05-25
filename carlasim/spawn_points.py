import random
import json
import numpy as np
import glob
import os
import sys
import time
import logging
from tqdm import tqdm
import cv2
try:
    sys.path.append(glob.glob('carla-0.9.7-py3.5-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla
import map
from collections import OrderedDict
from bounding_boxes import ClientSideBoundingBoxes
import configparser

zones = {
    1: ((0, 0), (3, 3)),
    2: ((0, 7), (3, 9)),
    3: ((0, 13), (3, 15)),
    4: ((1, 18), (3, 20)),
    5: ((1, 22), (2, 25)),
    6: ((4, 26), (6, 28)),
    7: ((13, 26), (14, 28)),
    8: ((14, 13), (14, 16)),
    9: ((13, 7), (14, 9)),
    10: ((13, 0), (14, 2)),
}
cameras = []
# pedestrians = []
pedestrians = {zone:[] for zone in range(10)}
view_width = 1280
view_height = 720
view_fov = 90
fps = 15
complete_cams = 0

def get_rand_in_zone(zone):
    x = random.randint(zones[zone][0][0], zones[zone][1][0])
    y = random.randrange(zones[zone][0][1], zones[zone][1][1])
    return x, y


def spawn_in_zone(world: carla.World, M: map.Map, zone: int, walker_bp: carla.ActorBlueprint):
    result = None
    while not result:
        point_index = get_rand_in_zone(zone)
        point_coords = M.translate_to_coord(point_index)
        spawn_point = carla.Transform(carla.Location(x=point_coords[0], y=point_coords[1], z=1),
                                    carla.Rotation(yaw=random.randrange(0, 360)))
        result = world.try_spawn_actor(walker_bp, spawn_point)
        
    return result, spawn_point


def camera_blueprint(world, cam_type='sensor.camera.rgb'):
    """
    Returns camera blueprint.
    """
    # camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp = world.get_blueprint_library().find(cam_type)
    camera_bp.set_attribute('image_size_x', str(view_width))
    camera_bp.set_attribute('image_size_y', str(view_height))
    camera_bp.set_attribute('fov', str(view_fov))
    camera_bp.set_attribute('sensor_tick', str(1 / fps))
    return camera_bp


def setup_camera(world, cam_id, location, writer, rotation, cam_type='rgb'):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        camera_transform = carla.Transform(location, rotation)
        if cam_type == 'rgb':
            camera = world.spawn_actor(camera_blueprint(world), camera_transform)
            camera.listen(lambda image: get_rgb_image(
                cam_id, writer, image))
            cameras.append(camera)
        calibration = np.identity(3)
        calibration[0, 2] = view_width / 2.0
        calibration[1, 2] = view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = view_width / \
            (2.0 * np.tan(view_fov * np.pi / 360.0))
        cameras[cam_id].calibration = calibration


def get_rgb_image(cam_id, writer, img):
    global pedestrians, complete_cams
    bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(
        pedestrians[cam_id], cameras[cam_id])
    image = ClientSideBoundingBoxes.process_img(
        img, view_width, view_height)
    # image = cv2.UMat(image)
    # print(len(pedestrians), len(bounding_boxes))
    if len(bounding_boxes) != 0:
        # box = bounding_boxes[0]
        for person, box in enumerate(bounding_boxes):
            box = np.delete(box, 2, 1)
            arr_box = np.asarray(box)
            height = abs(arr_box[0][1] - arr_box[4][1])
            width = abs(arr_box[0][0] - arr_box[3][0])
            coords = [arr_box[3], arr_box[4]]
            point = box.mean(0).getA().astype(int)
            if point[0][0] > view_width or point[0][1] > view_height or point[0][0] < 0 or point[0][1] < 0:
                continue
            # tracks[cam_id][img.frame] = (point[0][0], point[0][1], width, height, coords[0][0], coords[0][1], coords[1][0], coords[1][1])
            # * Crops box from the image and save image
            abs_left = point[0][0] - int(width / 2) - 5
            abs_top  = point[0][1] - int(height / 2) - 5
            # * Handle cases where box corners are out of image boundaries
            crop_left = max(0, abs_left)
            crop_top = max(0, abs_top)
            # * Adding + 10 as it takes the -5 from {abs_left} and {abs_top} into account 
            cropped = image[crop_top: min(view_height, int(abs_top + height + 10)), crop_left: min(view_width, int(abs_left + width + 10))]
            # TODO Should save this in a folder
            try:
                cv2.imwrite("images/cam_{}/person_{}.jpg".format(cam_id, person), cropped)
            except:
                print(cam_id, " ", person)
        complete_cams += 1


        
def main():
    config = configparser.ConfigParser()
    config.read("10_camera_config.ini")
    cam_count = int(config['GENERAL']['Cameras'])
    walkers_list = []
    all_id = []
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    M = map.Map()

    

    try:
        global pedestrians
        world = client.get_world()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 1 / 15
        world.apply_settings(settings)
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor



        # -------------
        # Spawn Walkers
        # -------------
        batch = []
        walker_speed = []
        blueprintsWalkers = [world.get_blueprint_library().find("walker.pedestrian.0001"),
                             world.get_blueprint_library().find("walker.pedestrian.0002"),
                             world.get_blueprint_library().find("walker.pedestrian.0003"),
                             world.get_blueprint_library().find("walker.pedestrian.0004"),
                             world.get_blueprint_library().find("walker.pedestrian.0005"),
                            #  world.get_blueprint_library().find("walker.pedestrian.0006"), This is used as model for simulation
                             world.get_blueprint_library().find("walker.pedestrian.0007"),
                             world.get_blueprint_library().find("walker.pedestrian.0008"),
                             world.get_blueprint_library().find("walker.pedestrian.0009"),
                             world.get_blueprint_library().find("walker.pedestrian.0010"),
                             world.get_blueprint_library().find("walker.pedestrian.0011"),
                             world.get_blueprint_library().find("walker.pedestrian.0012"),
                             world.get_blueprint_library().find("walker.pedestrian.0013"),
                             world.get_blueprint_library().find("walker.pedestrian.0014")]
        spawn_locations = []
        
        for zone in tqdm(zones.keys()):
            for bp in tqdm(blueprintsWalkers):
                walker, spawn_point = spawn_in_zone(world, M, zone, bp)
                pedestrians[zone-1].append(walker)
                walkers_list.append({"id": walker.id})
                spawn_locations.append(spawn_point)
                # print("Spawned person")

        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # pedestrians = world.get_actors().filter('walker.*')
        # print(len(pedestrians))
        # pedestrians = all_actors
        # print(len(pedestrians))
        # -------------
        # Spawn Cameras
        # -------------
        time.sleep(0.5)
        for camera in range(cam_count):
            cam = config['CAMERA_' + str(camera + 1)]
            location = carla.Location(x=float(cam['PosX']), y=float(
                cam['PosY']), z=float(cam['PosZ']))
            rotation = carla.Rotation(roll=float(cam['Roll']), pitch=float(
                cam['Pitch']), yaw=float(cam['Yaw']))
            writer = None
            setup_camera(world, camera, location, writer, rotation, cam_type='rgb')
            time.sleep(0.1)

        world.wait_for_tick()
        while complete_cams < 10:
        # time.sleep(10)
            world.wait_for_tick()
            # break

    finally:
        for camera in cameras:
            camera.destroy()
        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        # TODO Save locations in a format that can be easily imported and spawned in the simulation
        # print(spawn_locations)
        # import pickle
        # with open('spawn_locations.pkl', 'wb') as f:
        #     pickle.dump(spawn_locations, f)
        transforms = []
        for t in spawn_locations:
            # print(i.__dict__)
            transform = {
                "location": {
                    'x': t.location.x,
                    'y': t.location.y,
                    'z': t.location.z,
                },
                "rotation": {
                    'yaw': t.rotation.yaw
                }
            }
            transforms.append(transform)

        import json
        with open("locations.json", "w") as file:
            file.write(json.dumps(transforms))
        # print(transform)
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
