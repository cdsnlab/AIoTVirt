from microservices.microservice import Microservice
import numpy as np
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker


class Track(Microservice):
    def __init__(self, boxes: list = [], tracker: str = "IOUTracker", parent_kwargs: dict = {}):
        """[summary]

        Args:
            box_type (str, optional): Box definition type. Options are "center-width-height", 
                                      "topleft-width-height" and "topleft-botright"
                                      Defaults to "topleft-width-height".
            boxes (list, optional): ROI of Boxes to be cropped. Defaults to [].
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.tracker = self._init_tracker(tracker)
        self.boxes = boxes

    def _process(self, start_time, inputs: dict = {}):
        """[summary]
        Cropping with incoming boxes definition
        Args:
            imageid (float): ID of image in gallery
            image (np.array): Image to crop from
            boxes (list): Boxes to crop
        """
        # print(start_time, inputs)

        image_id, image, boxes = None, None, None
        if 'image_id' in inputs:
            image_id = inputs['image_id']
        if 'image' in inputs:
            image = inputs['image']
            image = np.array(image)
        if 'boxes' in inputs:
            boxes = inputs['boxes']
        if 'scores' in inputs:
            scores = inputs['scores']
        if not boxes:
            boxes = self.boxes

        class_ids = [1 for _ in scores]

        if image.any():
            tracks = self.tracker.update(boxes, scores, class_ids)
            self._sink(start_time, tracks)
        else:
            print("No image to track from")
            self._sink(start_time, None)

    def _init_tracker(self, tracker='CentroidTracker', max_lost=0, iou_threshold=0.4, min_detection_confidence=0.4, max_detection_confidence=0.7):
        if tracker == 'CentroidTracker':
            return CentroidTracker(
                max_lost=max_lost, tracker_output_format='mot_challenge')
        elif tracker == 'CentroidKF_Tracker':
            return CentroidKF_Tracker(
                max_lost=max_lost, tracker_output_format='mot_challenge')
        elif tracker == 'SORT':
            return SORT(
                max_lost=max_lost, tracker_output_format='mot_challenge', iou_threshold=iou_threshold)
        elif tracker == 'IOUTracker':
            return IOUTracker(
                max_lost=max_lost, iou_threshold=iou_threshold, min_detection_confidence=min_detection_confidence,
                max_detection_confidence=max_detection_confidence, tracker_output_format='mot_challenge')
