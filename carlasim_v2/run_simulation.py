import argparse
from tqdm import tqdm
import pygame
import pandas as pd
import os
from carla_server import SynchronousServer
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
        default = '/home/tung/carla/map_coor',
        help = 'Path to the folder that holds the trajectory files'
    )
    argparser.add_argument(
        '--data_dir',
        default = 'data',
        help = 'Path to the folder that contains simulation data'
    )
    argparser.add_argument(
        '-s', '--start',
        type = int,
        default = 1,
        help = 'The number of the FIRST track that this instance is supposed to handle.'
    )
    argparser.add_argument(
        '-r', '--end',
        type = int,
        help = 'The number of the LAST track that this instance is supposed to handle.'
    )
    argparser.add_argument(
        '-o', '--record_mode',
        type = int,
        default = 0,
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


def load_tracks(dir_path, log_file, start, end):
    tracks = []
    check = {}
    try:
        with open('data/run_logs/{}.txt'.format(log_file), 'r') as file:
            for line in file:
                check[line.split('\n')[0]] = 1
    except:
        pass
    for file in os.listdir(os.path.join(dir_path)):
        try:
            if check[file.split('.')[0]] == 1:
                pass
        except:
            track_num = file.split('.')[0].split('_')[0]
            if (int(track_num) >= start) & (int(track_num) <= end):
                track_df = pd.read_csv(os.path.join(dir_path, file), engine = 'python')
                track = track_df.values.tolist()
                tracks.append((file.split('.')[0], track))
            
    tracks.sort(key = lambda x:int(x[0].split('_')[0]))
    return tracks

def main():
    args = argparser()
    tracklets = load_tracks(args.tracks_dir, args.script_name, args.start, args.end)
    #pygame.init()

    server = SynchronousServer(args.data_dir, args.config_file, args.port, args.record_mode, args.script_name)
    
    try:
        for track_id, track in tracklets:

            # track_id = 'dummy'
            # track_id = track_id + '_' + str(index)
            # track = [(-68.2, -3.1, 1), (-63, -3.6, 1)]#[(-73.5, 61.8, 1), (-73.5, 75.8, 1), (-73.5, 85.8, 1)]#[(-68.2, -3.1, 1), (-51.2, -3.6, 1)]#[(-90.3,61.5, 1), (-87.6, 41.1, 1), (-90.3, 27.2, 1)]
            # server.game_loop(track_id, track)
            # break
            server.game_loop(track_id, track)
            

    
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
