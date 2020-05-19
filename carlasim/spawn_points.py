import random
import json
import numpy as np
import glob
import os
import sys
import time
import logging
from tqdm import tqdm
try:
    sys.path.append(glob.glob('carla-0.9.7-py3.5-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla
import map

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


def main():
    walkers_list = []
    all_id = []
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    M = map.Map()

    try:
        world = client.get_world()
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
                             world.get_blueprint_library().find("walker.pedestrian.0006"),
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
                walkers_list.append({"id": walker.id})
                spawn_locations.append(spawn_point)
                print("Spawned person")

        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # TODO Add cameras and crop images of people

        while True:
            world.wait_for_tick()

    finally:
        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        # TODO Save locations in a format that can be easily imported and spawned in the simulation
        print(spawn_locations)
        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
