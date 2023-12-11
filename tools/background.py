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


def detect_lines(white, yellow):
    w = cv2.Canny(white, 120, 150)
    lines = cv2.HoughLines(w, 1, math.pi / 180, 100)

    return lines, w

def detect(path):
    src = cv2.imread(path)
    h, w = src.shape[:2]
    
    white, yellow = filter_lanes(src, cvt=cv2.COLOR_GRAY2BGR)

    lines, c = detect_lines(white, yellow)

    print(len(lines))
    for line in lines:
        r, t = line[0]
        tx, ty = np.cos(t), np.sin(t)
        x0, y0 = r * tx, r * ty
        x1, y1 = int(x0 - w * ty), int(y0 + h * tx)
        x2, y2 = int(x0 + w * ty), int(y0 - h * tx)

        src = cv2.line(src, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    show_img = np.hstack([src, white, c])
    cv2.imshow('video', show_img)


if __name__ == '__main__':
    detect('ex.png')