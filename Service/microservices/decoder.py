from microservices.microservice import Microservice
from time import perf_counter, time
import cv2
import numpy as np


class Decoder(Microservice):
    def __init__(self, rtsp_url: str = None, decode_only: bool = True, output_shape: list = None,
                 transpose: list = None, opencv: bool = False, drop_frame: int = 2, parent_kwargs: dict = {}):
        """[summary]
​
        Args:
            rtsp_url (str): URL of RTSP source
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.url = rtsp_url
        
        self.decode_only = decode_only
        self.opencv = opencv
        self.is_started = False
        if "rtsp" in rtsp_url:
            gst = "rtspsrc location={url} latency=100 ! queue ! rtph264depay ! h264parse ! nvv4l2decoder drop-frame-interval={drop_frame} ! nvvidconv ! \
                    video/x-raw,width={width},height={height}, format=(string)BGRx ! videoconvert ! appsink sync=false".format(url=rtsp_url, drop_frame=drop_frame, width=output_shape[1], height=output_shape[0])
        else:
            # gst = "filesrc location={url} ! qtdemux name=demux demux. ! queue ! rtph264depay ! h264parse ! nvv4l2decoder drop-frame-interval={drop_frame} ! nvvidconv ! \
            #         video/x-raw,width={width},height={height}, format=(string)BGRx ! videoconvert ! appsink sync=false".format(url=url, drop_frame=drop_frame, width=output_shape[1], height=output_shape[0])
            gst = "filesrc location={url} ! decodebin ! nvvidconv ! video/x-raw,width={width},height={height}, format=(string)BGRx ! videoconvert ! appsink".format(
                url=rtsp_url, width=output_shape[1], height=output_shape[0])
        if not opencv:
            self.url = gst

        if self.decode_only:
            self.output_shape = None
            self.transpose = None
            self._process = self._decode
        else:
            self.output_shape = tuple(output_shape)
            self.transpose = transpose
            if opencv:
                self._process = self._resize_transpose_opencv
            else:
                self._process = self._resize_transpose_gst

    def start(self):
        # Ran only once if head of pipeline
        if self.is_started:
            return "Already running"
        if self.is_head:
            if self.opencv:
                self.cap = cv2.VideoCapture(self.url)
            else:
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_GSTREAMER)
            self.thread.start()
            self.is_started = True
        else:
            print("Can't use this method if Decoder isn't head of pipeline!")

    def _thread(self):
        """[summary]
        Microservice processing thread. Times execution of each iteration
        Shouldn't need to be overwritten
        """
        # while True:
        print("starting video: ", self.cap.isOpened())
        while self.cap.isOpened():
            # * Check run flag and stop if necessary
            if not self._RUN_THREAD:
                self.cap.release()
                break
            start = perf_counter()
            self._process()
            self.metrics["process"]["total"] += perf_counter() - start
            self.metrics["process"]["iters"] += 1

    def _decode(self):
        ret, frame = self.cap.read()
        start_time = time()
        if ret:
            self._sink(start_time, frame)
        else:
            self.cap.release()

    def _resize_transpose_opencv(self):
        readTime = perf_counter()
        ret, frame = self.cap.read()
        readTime = perf_counter() - readTime
        self.metrics["read"]["total"] += readTime
        self.metrics["read"]["list"].append(readTime)
        start_time = time()
        if ret:
            resTime = perf_counter()
            frame = cv2.resize(frame, self.output_shape)
            resTime = perf_counter() - resTime
            self.metrics["resize"]["total"] += resTime
            self.metrics["resize"]["list"].append(resTime)
            if self.links[0].is_local: #* here!
                # frame = frame.transpose(self.transpose).astype(np.uint8) #original
                frame = frame.astype(np.uint8)
            else:
                _, frame = cv2.imencode(".jpg", frame) 
            self._sink(start_time, frame)
        else:
            self.cap.release()

    def _resize_transpose_gst(self):
        readTime = perf_counter()
        ret, frame = self.cap.read()
        readTime = perf_counter() - readTime
        self.metrics["read"]["total"] += readTime
        self.metrics["read"]["list"].append(readTime)
        start_time = time()
        if ret:
            if self.links[0].is_local:
                frame = frame.transpose(self.transpose).astype(np.float32)
            else:
                _, frame = cv2.imencode(".jpg", frame)
            self._sink(start_time, frame)
        else:
            self.cap.release()

    def kill_thread(self, timeout=2):
        """[summary]
        Stop the processing thread

        Args:
            timeout (int, optional): Timeout to wait for joining thread. Defaults to 2.
        """
        self._RUN_THREAD = False
        if self.thread.is_alive():
            self.cap.release()
            self.thread.join(timeout)
            print("Killed thread of service: ", self.name)
        else:
            print("Thread not running")

    def _metrics_setup(self):
        super()._metrics_setup()
        self.metrics["resize"] = {
            "help": 1,
            "total": 0,
            "list":  []
        }
        self.metrics["read"] = {
            "help": 0,
            "total": 0,
            "list": []
        }
