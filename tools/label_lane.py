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
    


def on_mouse(clicked_list: list, state: dict):
    def _on(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            clicked_list.append((state['key'], (x, y)))
        elif event == cv2.EVENT_MBUTTONUP:
            if len(clicked_list) > 0:
                pos = np.array([x, y])
                nearest = min(clicked_list, key=lambda e: np.sum((np.array(e[1]) - pos)**2))
                clicked_list.remove(nearest)
    return _on

def main(path: str, lane_file: str = None):
    cv2.namedWindow('frame')
    state = {'key': 'w'}

    frame = cv2.imread('tools/lane/ex.png')


    clicked = []
    
    if lane_file is not None:
        with open(lane_file, 'r') as f:
            for line in f:
                k, x, y = line.split()
                clicked.append((k, (int(x), int(y))))

    cv2.setMouseCallback('frame', on_mouse(clicked, state))

    tmp_dict = dict()
    while True:
        img = frame.copy()
        cv2.putText(img, f'key={state["key"]}', org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 0, 0))

        for e in clicked:
            key, pos = e
            x, y = pos
            cv2.circle(img, pos, 5, (255, 0, 0), 2)
            if key in tmp_dict:
                cv2.line(img, pos, tmp_dict[key], color=(255,0,0), thickness=1)
            tmp_dict[key] = pos

            c = None
            x -= 5
            y += 5
            if key == 'w':
                c = '^'
                y -= 8
            elif key == 'a':
                c = '<'
                x -= 8
            elif key == 'd':
                c = '>'
                x += 8
            elif key == 's':
                c = 'v'
                y += 8
            if c is not None:
                cv2.putText(img, c, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=2)

        cv2.imshow('frame', img)
        ch = cv2.waitKey(1)
        if ch == 27:
            break
        elif ch in [ord(c) for c in 'qweasd']:
            state['key'] = chr(ch)
        elif ch == ord('z'):
            if len(clicked) > 0:
                clicked.pop()
        elif ch == ord('`'):
            print(clicked)
            with open(os.path.join(os.path.dirname(__file__), 'lane.txt'), 'w') as f:
                for e in clicked:
                    k, (x, y) = e
                    f.write(f'{k} {x} {y}\n')




if __name__ == '__main__':
    main('test.avi', os.path.join(os.path.dirname(__file__), 'lane.txt'))
