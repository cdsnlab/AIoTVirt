import cv2
import numpy as np
import math


def average(path):
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    acc = np.zeros((int(height), int(width), 3), dtype=np.float32)
    acc2 = np.zeros((int(height), int(width), 3), dtype=np.float32)
    
    frame_id = 0
    
    frames = []
    while True:
        ret_val, frame = cap.read()

        if ret_val:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 3600*frame_id)
            frame_id += 1
            frames.append(frame)
            print(f'Captured {frame_id}')
        else:
            break
    