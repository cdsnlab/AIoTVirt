from microservices.microservice import Microservice
import numpy as np



class Crop(Microservice):
    def __init__(self, box_type: str = "topleft-width-height", boxes: list = [], parent_kwargs: dict = {}):
        """[summary]

        Args:
            box_type (str, optional): Box definition type. Options are "center-width-height", 
                                      "topleft-width-height" and "topleft-botright"
                                      Defaults to "topleft-width-height".
            boxes (list, optional): ROI of Boxes to be cropped. Defaults to [].
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.boxes = boxes
        if box_type == "center-width-height":
            self._crop = self._process_cwh
        elif box_type == "topleft-width-height":
            self._crop = self._process_tlwh
        elif box_type == "topleft-botright":
            self._crop = self._process_tlbr

        if not self.is_local:
            self.gallery = {}

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
        if not boxes:
            boxes = self.boxes

        if image.any():
            self._crop(start_time, image, boxes)
        else:
            image = self.gallery[image_id]
            self._crop(start_time, image, boxes)

    def _process_cwh(self, start_time, input, boxes):
        output = []
        for center_x, center_y, width, height in boxes:
            output.append(input[:, :, center_y - width // 2:center_y + width // 2,
                          center_x - height // 2:center_x + height // 2])

        self._sink(start_time, output)

    
    def _process_tlwh(self, start_time, input, boxes):
        output = []
        for top, left, height, width in boxes:
            result = input[:, :, left:left + width, top:top + height]
            if result.size != 0:
                output.append(result[0].astype(np.uint8))

        self._sink(start_time, output)

    def _process_tlbr(self, start_time, input, boxes):
        output = []
        for top, left, bottom, right in boxes:
            output.append(input[left:right, top:bottom])

        self._sink(start_time, output)

    # def prepare(self, input):
    #     if len(input) == 2:
    #         # * Adding an image to the gallery
    #         # print(input)
    #         self.gallery[input["start_time"]] = input["inputs"]["image"]
    #     else:
    #         # * Process image from gallery
    #         self.queue.put(list(input.values()))
