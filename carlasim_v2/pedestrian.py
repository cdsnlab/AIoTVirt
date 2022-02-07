from collections import OrderedDict
import os
import csv
import carla

class Pedestrian(object):
    def __init__(
        self,
        track_id, 
        walking_track,
        height,
        carla_id, 
        carla_actor,
        ai_controller,
        img_dir_path,
        result_dir_path,
        log_dir_path,
        num_cams,
        first_appear_frame,
    ):
        self.id = track_id
        self.walking_track = walking_track
        self.height = height
        self.curr_frame = first_appear_frame
        self.carla_actor = carla_actor
        self.carla_id = carla_id
        self.ai_controller = ai_controller
        self.ai_controller.start()
        self.save_track_path = os.path.join(result_dir_path, track_id + '.csv')
        self.img_dir_path = os.path.join(img_dir_path, track_id)
        if not os.path.exists(self.img_dir_path):
            os.mkdir(self.img_dir_path)
        
        self.log_path = os.path.join(log_dir_path, track_id + '.txt')

        self.num_cams = num_cams
        self.recorded_track = dict()
        for cam_id in range(self.num_cams):
            self.recorded_track[cam_id] = []

        next_loc = self.walking_track.pop(0)
        self.next_location = carla.Location(next_loc[0], next_loc[1], self.height)
        self.speed = next_loc[2]
        self.elapsed_frame = 0
        self.logs = ''

    def save_track(self):
        with open(self.save_track_path, mode='w') as file:
            writer = csv.writer(file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

            headers = ['Camera {}'.format(cam) for cam in range(self.num_cams)]
            writer.writerow(['#'] + headers)
            # tracks = {cam: list(tr.values())
            #             for cam, tr in self.tracks.items()}
            
            print(self.id, [len(self.recorded_track[i]) for i in range(self.num_cams)])

            for frame in range(len(self.recorded_track[0])):
                row = [self.recorded_track[cam][frame] for cam in range(self.num_cams)]
                writer.writerow([int(frame)] + row)

        with open(self.log_path, mode = 'w') as file:
            file.write(self.logs)
            file.close()

    def get_next_location(self):
        next_loc = self.walking_track.pop(0)
        self.next_location = carla.Location(next_loc[0], next_loc[1], self.height)
        self.speed = next_loc[2]

    def arrive_at_next_location(self):
        current_location = self.ai_controller.get_location()
        if current_location.distance(self.next_location) < 3:
            return True
        return False

    def walk(self):
        self.ai_controller.go_to_location(self.next_location)
        self.ai_controller.set_max_speed(self.speed)

    def logging(self, log_msg):
        self.logs += log_msg
        self.logs += '\n'

    def complete_track(self):
        self.ai_controller.stop()
        self.carla_actor.destroy()
        self.save_track()
