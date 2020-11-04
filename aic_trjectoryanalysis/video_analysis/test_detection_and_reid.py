import argparse, sys

import cv2
import torch
import time
sys.path.insert(1,"../../../mmdetection/")
from mmdet.apis import inference_detector, init_detector, get_cropped_objects


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--detector', default='yolo', help='yolo, ssd, faster rcnn')
    parser.add_argument('--config', default='../../../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py', help='test config file path')
    parser.add_argument('--checkpoint', default = '../../../mmdetection/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco-139f5633.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.6, help='bbox score threshold')
    args = parser.parse_args()
    # if args.detector=="yolo":
    #     args.config = '../../../mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
    #     args.checkpoint = '../../../mmdetection/checkpoints/yolo/yolov3_d53_mstrain-608_273e_coco-139f5633.pth'
    # elif args.detector =="ssd":
    #     args.config = '../../../mmdetection/configs/ssd/ssd512_coco.py'
    #     args.checkpoint = "../../../mmdetection/checkpoints/ssd/ssd512_coco_20200308-038c5591.pth"
    # elif args.detector == "faster_rcnn":
    #     args.config = '../../../mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    #     args.checkpoint = "../../../mmdetection/checkpoints/faster_rcnn/faster_rcnn_x101_64x4d_fpn_2x_coco_20200512_161033-5961fa95.pth"

    # print(args)
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)
    #camera = cv2.VideoCapture(args.camera_id)
    videofile = "/home/spencer1/samplevideo/AIC20_track1_vehicle_counting/AIC20_track1/Dataset_A/cam_5.mp4"
    #videofile = "/home/spencer1/samplevideo/AIC20_track3_MTMC/train/S01/c005/vdo.avi"
    camera = cv2.VideoCapture(videofile)

    print('Press "Esc", "q" or "Q" to exit.')
    # while True:
    #     ret_val, img = camera.read()
    #     start_time = time.time()
    #     result = inference_detector(model, img)
    #     #print(result)
    #     end_time = time.time()
    #     ch = cv2.waitKey(1)
    #     if ch == 27 or ch == ord('q') or ch == ord('Q'):
    #         break
    #     print("exec time: {}".format(end_time-start_time))
    #     print(result[0])
    #     model.show_result(
    #         img, result, score_thr=args.score_thr, wait_time=1, show=True)

    #* spencer's cropping instances
    while True:
        ret_val, img = camera.read() #960, 1280
        
        start_time = time.time()
        #TODO properly get the class labels...
        list_of_cropped_items = get_cropped_objects(model, img) #! 2: car, 7: van, 9: traffic light
        print(list_of_cropped_items)
        end_time = time.time()        

if __name__ == '__main__':
    main()