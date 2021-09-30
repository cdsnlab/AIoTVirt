from microservices.microservice import Microservice
from microservices.trtclient_http import Tritonclient
from time import perf_counter
import cv2
import numpy as np

class Inference(Microservice):
    def __init__(self, model: str, infer_url: str, separate_requests: bool = False, concurrency: int = 2, parent_kwargs: dict = {}):
        """[summary]
        Args:
            model (str): Name of DNN model
            infer_url (str): URL for infer request
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.model = model
        self.infer_url = infer_url
        self.concurrency = concurrency
        self.tritonclient = Tritonclient(
            self.model, self.infer_url, self.concurrency)
        self.separate_requests = separate_requests
        self.prepare_queue = self._setup_queue("prepare_queue", 100)
        if not self.is_local:
            self.prepare_thread = self._spawn_thread(target=self.preprocess, daemon=True)
            self.prepare_thread.start()
        if separate_requests:
            self.infer_thread = self._spawn_thread(
                target=self._process_async, args=(), daemon=True)
            self._process = self._create_requests
            self._infer_queue = self._setup_queue("infer_queue", 100)
            self.infer_thread.start()
        else:
            self._process = self._process_normal

        self.threads = []

    def _create_requests(self, start_time, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        requests = self.tritonclient.prepare_requests(np.array(inputs))
        self._infer_queue.put({
            "start_time": start_time,
            "inputs": requests,
            "og_inputs": inputs
        })

    def _process_async(self):
        while True:
            # * Check run flag and stop if necessary
            if not self._RUN_THREAD:
                break
            request_inputs = []
            request_outputs = []
            request_starts = []
            og_inputs = []
            for _ in range(self.concurrency):
                request = self._infer_queue.get()
                request_inputs.append(request["inputs"][0])
                request_outputs.append(request["inputs"][1])
                request_starts.append(request["start_time"])
                og_inputs.append(request["og_inputs"])
            start = perf_counter()

            results = self.tritonclient.process_async(
                request_inputs, request_outputs)
            infertime = perf_counter() - start
            self.metrics["inference"]["total"] += infertime
            self.metrics["inference"]["list"].append(infertime)

            for start, result, og_input in zip(request_starts, results, og_inputs):
                output = {'inputs': og_input, 'result': result}
                self._sink(start, output)

    def _process_normal(self, start_time, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        result = self.tritonclient.infer(inputs)
        output = {'inputs': inputs, 'result': result}
        self._sink(start_time, output)

    def prepare(self, inputs):
        """[summary]
        Called only when from remote. Web server passes decoded object.
        Stored in processing queue
        Args:
            inputs (object): Input object
        """
        self.prepare_queue.put(inputs)

    def preprocess(self):
        self.metrics["decode"] = {
            "help": "JPG Decode time",
            "total": 0,
            "list": []
        }
        while True:
            # * Check run flag and stop if necessary
            if not self._RUN_THREAD:
                break
            start = perf_counter()
            inputs = self.prepare_queue.get()
            enc = inputs["inputs"]
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR).transpose(
                (2, 0, 1)).astype(np.float32)
            inputs["inputs"] = dec
            self.queue.put(inputs)
            timeTaken = perf_counter() - start
            self.metrics["decode"]["total"] += timeTaken
            self.metrics["decode"]["list"].append(timeTaken)

    def _metrics_setup(self):
        super()._metrics_setup()
        self.metrics["inference"] = {
            "help": "Inference metrics for reID. Total time taken, number of iterations and per batch process times.",
            "total": 0,
            "iters": 0,
            "list": []
        }
