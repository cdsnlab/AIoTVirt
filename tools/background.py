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
    
    print(f'Accumulating {len(frames)} frames')
    pv = 1 / len(frames)
    for frame in frames:
        cv2.accumulate(pv * frame.astype(np.float32), acc)
        
        filtered, _ = filter_lanes(frame, cv2.COLOR_GRAY2BGR)
        cv2.accumulate(pv * filtered.astype(np.float32), acc2)

    
    show_acc = cv2.GaussianBlur(acc, (9, 9), 1.0).astype(np.uint8)
    cv2.imwrite('ex.png', show_acc)
    cv2.imwrite('ex2.png', acc2.astype(np.uint8))

def filter_lanes(img, cvt=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 180), (180, 32, 255))
    yellow = cv2.inRange(hsv, (5, 30, 100), (70, 255, 255))

    if cvt is not None:
        white = cv2.cvtColor(white, cvt)
        yellow = cv2.cvtColor(yellow, cvt)
    return white, yellow
    

def detect(path):
    src = cv2.imread(path)
    h, w = src.shape[:2]
    
    white, yellow = filter_lanes(src, cvt=cv2.COLOR_GRAY2BGR)

    cv2.imshow('white', white)
    cv2.imshow('yellow', yellow)


if __name__ == '__main__':
    detect('ex.png')