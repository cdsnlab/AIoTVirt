from microservices.microservice import Microservice
from microservices.post_detectnet import Post_DetectNet


class Postprocess(Microservice):
    def __init__(self, postprocess_type: str, post_process_kwargs: dict = {}, parent_kwargs: dict = {}):
        """[summary]
        Args:
            postprocess_type (str): Name of Inference type
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.postprocess_type = postprocess_type
        if self.postprocess_type == 'DetectNet':
            self._process = self._detectnet_process
            self.pdn = Post_DetectNet(**post_process_kwargs)

    def _detectnet_process(self, start_time, inputs, visualize=False):
        input = inputs['result']
        frame = inputs['inputs']
        coordinates, conf_scores = self.pdn.postprocess(input, frame, visualize)
        output = {'boxes': coordinates, 'image_id': 0, 'image': frame, 'scores': conf_scores}
        self._sink(start_time, output)
