import argparse
from tqdm import tqdm
import pygame
import pandas as pd
import os
from carla_server_for_fig import SynchronousServer
from client_carla import ClientSideBoundingBoxes as CameraClient

def argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar = 'P',
        default = 2000,
        type = int,
        action = 'store',
        help = 'TCP port to listen to (default: 2000)'
    )
    argparser.add_argument(
        '--config_file',
        default = 'camera_config.ini',
        help = 'Path to the configuration file'
    )
    argparser.add_argument(
        '--tracks_dir',
        default = 'carla_coor_for_fig',
        help = 'Path to the folder that holds the trajectory files'
    )
    argparser.add_argument(
        '--data_dir',
        default = 'data',
        help = 'Path to the folder that contains simulation data'
    )
    argparser.add_argument(
        '--num_walkers',
        default = 50,
        type = int,
        help = 'The maximum num of pedestrians in the simulation at any moment.'
    )
    argparser.add_argument(
        '--start',
        type = int
    )
    argparser.add_argument(
        '--end',
        type = int
    )
    argparser.add_argument(
        '-o', '--record_mode',
        type = int,
        default = 3,
        help = 'This decides whether we need the images. (0) nothing, (1) only rbg images, (2) only semseg images, (3) both rgb and semseg images.'
    )
    argparser.add_argument(
        '--script_name',
        metavar = 'SCRIPT NAME',
        type = str,
        help = 'The name of the bash script, used for logging.'
    )
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    args = argparser.parse_args()
    return args

def read_last_saved(file_name):
    try:
        with open('data/run_logs/{}.txt'.format(file_name), 'r') as file:
            for line in file:
                pass
            last_saved = line
        return int(last_saved)
    except:
        return -1

def load_tracks(dir_path, log_file, start = 2, end = 61):
    tracks = []
    check = {}

    
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        track_id = file_name.split('.')[0]
        
        track_df = pd.read_csv(file_path, engine = 'python')
        track = track_df.values.tolist()
        tracks.append((track_id, track))

    return tracks

def main():
    args = argparser()
    tracklets = load_tracks(args.tracks_dir, args.script_name, args.start, args.end)
    server = SynchronousServer(args.data_dir, args.config_file, args.port, args.record_mode, args.script_name)
    
    try:
        # for index, (track_id, track) in enumerate(tracklets):

        #     # track_id = 'dummy'
        #     # track_id = track_id + '_' + str(index)
        #     # track = [(-68.2, -3.1, 1), (-63, -3.6, 1)]#[(-73.5, 61.8, 1), (-73.5, 75.8, 1), (-73.5, 85.8, 1)]#[(-68.2, -3.1, 1), (-51.2, -3.6, 1)]#[(-90.3,61.5, 1), (-87.6, 41.1, 1), (-90.3, 27.2, 1)]
        #     # server.game_loop(track_id, track)
        #     # break
        #     server.game_loop(track_id, track)
        server.game_loop(tracklets, CONCURRENT_TRACKS = args.num_walkers)

    
    finally:
        for rgb_camera, semseg_camera in zip(server.rgb_camera_list, server.semseg_camera_list):
            rgb_camera[1].destroy()
            semseg_camera[1].destroy()
        print('[INFO] Cameras Destroyed!')

        #pygame.quit()



if __name__ == '__main__':
    main()

# for 
#         with SynchMode(world, server.rgb_camera_list, server.semseg_camera_list, server.fix_delta) as sync_mode:
#             clock.tick()
#             snapshot, rgb_image, semseg_image = sync_mode.tick(timeout = 2.0)
#             semseg_image.convert(carla.ColorConverter.CityScapesPalette)
#             image = CameraClient.process_img(semseg_image, server.view_width, server.view_height)
#             cv2.imwrite(os.path.join('images', 'test.jpg'), image)
