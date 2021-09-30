import cv2
import numpy as np

class Post_DetectNet():
    def __init__(self, img_size: list, num_class: int, threshold=0.5, box_norm=35):
        # -------------- MODEL PARAMETERS FOR DETECTNET_V2 --------------------------------
        self.model_h = img_size[0]
        self.model_w = img_size[1]
        self.NUM_CLASSES = num_class
        self.threshold = threshold

        stride = 16
        self.box_norm = box_norm

        self.grid_h = int(self.model_h / stride)
        self.grid_w = int(self.model_w / stride)
        self.grid_size = self.grid_h * self.grid_w

        self.grid_centers_w = []
        self.grid_centers_h = []
        self.x1_idx = 2 * 4 * self.grid_size
        self.y1_idx = self.x1_idx + self.grid_size
        self.x2_idx = self.y1_idx + self.grid_size
        self.y2_idx = self.x2_idx + self.grid_size

        for i in range(self.grid_h):
            value = (i * stride + 0.5) / self.box_norm
            self.grid_centers_h.append(value)

        for i in range(self.grid_w):
            value = (i * stride + 0.5) / self.box_norm
            self.grid_centers_w.append(value)

    def applyBoxNorm(self, o1, o2, o3, o4, x, y):
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid

        Returns:
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """
        o1 = (o1 - self.grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - self.grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + self.grid_centers_w[x]) * self.box_norm
        o4 = (o4 + self.grid_centers_h[y]) * self.box_norm
        return o1, o2, o3, o4

    def worker(self, c, h, bbs, scores, keepcounts, min_confidence, boxes):
        hw = h * self.grid_w
        for w in range(self.grid_w):
            i = w + hw
            score = keepcounts[c * self.grid_size + i]
            if score >= min_confidence:
                o1 = boxes[self.x1_idx + i]
                o2 = boxes[self.y1_idx + i]
                o3 = boxes[self.x2_idx + i]
                o4 = boxes[self.y2_idx + i]

                o1, o2, o3, o4 = self.applyBoxNorm(o1, o2, o3, o4, w, h)

                xmin = int(o1)
                ymin = int(o2)
                xmax = int(o3)
                ymax = int(o4)
                bbs.append([xmin, ymin, xmax - xmin, ymax - ymin])
                scores.append(float(score))

    def decode(self, outputs, min_confidence):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        bbs = []
        # class_ids = []
        scores = []
        # for c in analysis_classes:
        #     if c != 2:
        #         continue
        boxes = outputs[0]
        keepcounts = outputs[1]

        for h in range(self.grid_h):
            # self.worker(c, h, bbs, scores, keepcounts, min_confidence, boxes)
            self.worker(2, h, bbs, scores, keepcounts, min_confidence, boxes)

        return bbs, scores

    def postprocess(self, results, frame=None, visualize=False):
        detection_out = results['conv2d_bbox']
        keepCount_out = results['conv2d_cov/Sigmoid']

        results = self.decode(
            [detection_out.ravel(), keepCount_out.ravel()], self.threshold
        )
        coordinates = self.coordination(results)
        if visualize:
            return coordinates, self.visualizer(frame, coordinates)
        else:
            return coordinates

    def coordination(self, results):
        bboxes, scores = results
        indexes = cv2.dnn.NMSBoxes(bboxes, scores, self.threshold, 0.5)
        coordinates = []
        for idx in indexes:
            idx = int(idx)
            xmin, ymin, w, h = bboxes[idx]
            coordinates.append([xmin, ymin, w, h])
        return coordinates, scores

    def visualizer(self, image, coordinates):
        image_cpy = image.copy()
        color = [255, 0, 255]
        for coord in coordinates:
            xmin, ymin, w, h = coord
            # class_id = class_ids[idx]
            # color = [255, 0, 0] if class_id else [0, 0, 255]
            cv2.rectangle(image_cpy, (xmin, ymin), (xmin + w, ymin + h), color, 2)
        return image_cpy
