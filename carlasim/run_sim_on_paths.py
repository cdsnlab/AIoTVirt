import json
import server_bboxes_sim
import time
import argparse
from tqdm import tqdm

connections = {
    1: [2,3,4,5,6,7,8,9],
    2: [1,4,5,6,7,8,9],
    3: [1,2,6,7,8,9],
    4: [1,2,6,7,8,9],
    5: [1,2,6,7,8,9],
    6: [1,2,3,4,7,8,9],
    7: [1,2,3,4,5,6,9],
    8: [1,2,3,4,5,6,9],
    9: [1,2,3,4,5,6,7]
}

argparser = argparse.ArgumentParser(
    description=__doc__)
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '-s', '--startzone',
    metavar='s',
    default=1,
    type=int,
    help='Start zone (default: 1)')
# argparser.add_argument(
#     '-i', '--id',
#     metavar='i',
#     default=1,
#     type=int,
#     help='Start zone (default: 1)')
argparser.add_argument(
    '-ez', '--endzone',
    default=9,
    type=int,
    help='End zone (default: 2)')

args = argparser.parse_args()
paths = None

with open("paths_4x4.json", "r") as f:
    paths = json.load(f)

endzones = [3]
for zone in tqdm(endzones):
    zone = int(zone)
    counter = 0
    tracks = paths["{}-{}".format(args.startzone, zone)]
    for path in tqdm(tracks):
        start = time.time()
        client = server_bboxes_sim.BasicSynchronousClient()
        client.parse_config()
        client.connect_client(args.port)
        client.game_loop(path, args.startzone, zone, counter)
        counter += 1
        print(time.time() - start)
        time.sleep(1)
        break
