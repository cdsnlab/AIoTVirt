import cv2
import numpy as np
import os


def get_frame(path) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            return frame
    
def main(path: str, lane_file: str = None):
    cv2.namedWindow('frame')
    state = {'key': 'w'}

    frame = cv2.imread('tools/lane/ex.png')

    while True:
        img = frame.copy()
        cv2.imshow('frame', img)
        ch = cv2.waitKey(1)

    


if __name__ == '__main__':
    main('test.avi', os.path.join(os.path.dirname(__file__), 'lane.txt'))
