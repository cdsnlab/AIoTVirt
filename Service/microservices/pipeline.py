from .microservice import Microservice
from .util import ms_from_str
from typing import List


class Pipeline(object):  # Probably inherits from Thread/Process
    def __init__(self, pipeline_request: dict):
        # for key, val in microservices:
        #     val.
        # self.head = ...
        # for src, sink in links:
        #     src.link(sink)
        # * Force comparison with other pipelines in order to join if possible
        pass

    def _parse(self, pipeline_request: dict):
        
        # ! Request should look like:
        request = {
            "Decoder": {
                "rtsp_url": "rtsp://cdsnlab:CdsnLab7759@143.248.55.237:28556/ds-test",
                "common": {
                    "name": "decoder1",
                    "is_sink":  False,
                    "is_local":  True,
                    "is_head": True,
                } # ! These are not necessary. Just gives the outline
            },
            "Resize": {
                "source": "decoder1",
                "output_shape": [272,480],
                "dtype": "np.float32",
                "common": {}
            }
        }
        for ms_name, args in pipeline_request:
            try:
                parent_kwargs = args.popitem("common")
                # Used for parent class
            except KeyError:
                pass
            try:
                source = args.popitem("source")
                # TODO use for linking
                # ! Think about changing to instead be output 
            except KeyError:
                pass
            # ? Change Microservice child args to kwargs, so we pass dict directly?
            ms = ms_from_str(ms_name, args, parent_kwargs)

    def _compare(self, pipeline: Pipeline):
        # ?
        pass

    def _check_left_join():
        # TODO
        # Compare against all pipelines for possible joins
        # And create a left join with an already existing one
        pass

    def run():
        pass
        # self.head.start()
