from microservices.microservice import Microservice
import numpy as np
import cv2


class Resize(Microservice):
    def __init__(self, output_shape: list = [], dtype: str = None, parent_kwargs: dict = {}):
        """[summary]

        Args:
            output_shape (list): Shape of image after resize
            dtype (str): Data type of numpy array. Follows np.dtype options
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.output_shape = tuple(output_shape)
        self.dtype = np.dtype(eval(dtype))

    def _process(self, start_time, inputs):
        output = cv2.resize(inputs, self.output_shape).astype(self.dtype)
        self._sink(start_time, output)
