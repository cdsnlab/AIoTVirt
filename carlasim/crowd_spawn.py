import random
import glob
import sys
import map
from tqdm import tqdm
from collections import OrderedDict
from bounding_boxes import ClientSideBoundingBoxes 
sys.path.append(glob.glob('carla-0.9.7-py3.5-linux-x86_64.egg')[0])
import carla


class CrowdSpawn(object):
    def __init__(self, world):
        self.M = map.Map()
        self.world = world
        self.walkers_list = []
        self.blueprintsWalkers = [
            self.world.get_blueprint_library().find("walker.pedestrian.0001"),
            self.world.get_blueprint_library().find("walker.pedestrian.0002"),
            self.world.get_blueprint_library().find("walker.pedestrian.0003"),
            self.world.get_blueprint_library().find("walker.pedestrian.0004"),
            self.world.get_blueprint_library().find("walker.pedestrian.0005"),
            self.world.get_blueprint_library().find("walker.pedestrian.0007"),
            self.world.get_blueprint_library().find("walker.pedestrian.0008"),
            self.world.get_blueprint_library().find("walker.pedestrian.0009"),
            self.world.get_blueprint_library().find("walker.pedestrian.0010"),
            self.world.get_blueprint_library().find("walker.pedestrian.0011"),
            self.world.get_blueprint_library().find("walker.pedestrian.0012"),
            self.world.get_blueprint_library().find("walker.pedestrian.0013"),
            self.world.get_blueprint_library().find("walker.pedestrian.0014")]
        self.zones = {
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
        
        
    def get_rand_in_zone(self, zone):
        x = random.randint(self.zones[zone][0][0], self.zones[zone][1][0])
        y = random.randrange(self.zones[zone][0][1], self.zones[zone][1][1])
        return x, y


    def spawn_in_zone(self, zone: int, walker_bp: carla.ActorBlueprint):
        result = None
        while not result:
            point_index = self.get_rand_in_zone(zone)
            point_coords = self.M.translate_to_coord(point_index)
            spawn_point = carla.Transform(carla.Location(x=point_coords[0], y=point_coords[1], z=1),
                                        carla.Rotation(yaw=random.randrange(0, 360)))
            result = self.world.try_spawn_actor(walker_bp, spawn_point)
            
        return result, spawn_point
    
    
    def spawn_people(self, all_id):
        for zone in tqdm(self.zones.keys()):
            for bp in self.blueprintsWalkers:
                walker, _ = self.spawn_in_zone(zone, bp)
                self.walkers_list.append({"id": walker.id})

        for i in range(len(self.walkers_list)):
            all_id.append(self.walkers_list[i]["id"])
        # all_actors = self.world.get_actors(all_id)
        return all_id
