from microservices.microservice import Microservice
import cv2


class Transpose(Microservice):
    def __init__(self, transpose: list, parent_kwargs: dict = {}):
        """[summary]

        Args:
            transpose (list): Transpose as list argument
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.transpose = transpose

    def _process(self, start_time, inputs):
        output = inputs.transpose(self.transpose)
        self._sink(start_time, output)
