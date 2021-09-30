from microservices.microservice import Microservice
from microservices.trtclient_http import Tritonclient as httpClient
from microservices.trtclient_grpc import Tritonclient as grpcClient
from microservices.torchreid_osnet import FeatureExtractor
from microservices.torchreid_distance import compute_distance_matrix
from time import perf_counter, time
from torch import Tensor
import numpy as np
from microservices.util import resize_numpy

class Reid(Microservice):
    def __init__(self, model: str, device: str, engine: str, gallery=[[[0, 0, 0]]], batch: int = 1,
                 feature_extract: bool = True, metric: str = 'cosine', triton_args: dict = {}, grpc: bool = True, parent_kwargs: dict = {}):
        """[summary]
        Args:
            model (str): Name of reid feature extractor model
            device (str): 'cpu' or 'cuda'
            engine (str): 'triton' or 'torchreid'
            gallery (torch.Tensor or list of np.array, optional): Batch of gallery images. Defaults
            batch (int) : minumum size for cache to evaluate distance comparison

            for 'torchreid' engine:
                feature_extract (bool, optional): True if inputs need feature extraction. Defaults to True
                metric (str): Distance comparison metric, either 'euclidean' or 'cosine'. Defaults to cosine
            for 'triton' engine:
                triton_args (dict) : 'url', 'concurrency'
            parent_kwargs (dict, optional): Arguments for parent microservice. Defaults to {}.
        """
        super().__init__(**parent_kwargs)
        self.model = model
        self.device = device
        self.metric = metric
        self.batch = batch
        self.engine = engine
        self.cache = []
        self.batch_ids = []
        self.pipeline_times = []
        if self.engine == 'torchreid':
            self._init_torch(model, device, gallery, feature_extract)
        else:
            self._init_triton(triton_args, gallery, grpc)

    def _init_torch(self, model, device, gallery, feature_extract):
        self.extractor = FeatureExtractor(model_name=model, device=device)
        if not isinstance(gallery, Tensor):
            if isinstance(gallery, list):
                gallery = np.array(gallery, dtype=np.uint8)
            self.gallery = self.extractor(gallery)
        else:
            self.gallery = gallery
        if feature_extract:
            self._process = self._process_feature
        else:
            self._process = self._process_normal

    def _init_triton(self, triton_args, gallery, grpc):
        self.url = triton_args['url']
        self.concurrency = triton_args['concurrency']
        # * Extractor is triton client
        if grpc:
            self.extractor = grpcClient(self.model, self.url)
            # * Get infer shape from model metadata
            self.infer_shape = tuple(self.extractor.triton_client.get_model_metadata(
                self.model).inputs[0].shape[2:])
        else:
            self.extractor = httpClient(self.model, self.url, self.concurrency)
            # * Get infer shape from model metadata
            self.infer_shape = tuple(self.extractor.triton_client.get_model_metadata(
                self.model)["inputs"][0]["shape"][2:])
        
        # if isinstance(gallery, list):
        #     gallery = np.array(gallery, dtype=np.float32)

        gallery = np.zeros((1, 3, 64, 64), dtype=np.float32)

        self.gallery = list(self.extractor.infer([gallery]).values())[0]
        if self.engine == "triton":
            self.processed = True
            self.start = None
            self._process = self._process_triton
        elif self.engine == "async":
            self._init_triton_async()

    def _init_triton_async(self):
        self.infer_thread = self._spawn_thread(target=self._process_async, daemon=True)
        self._process = self._create_requests
        self._infer_queue = self._setup_queue("infer_queue", 100)
        self.infer_thread.start()

    def _process_feature(self, start_time, inputs):
        self.cache += inputs
        if len(self.cache) < self.batch:
            return None
        else:
            output = self.extractor(self.cache)
            output = self.compare(output, self.gallery)
            self._sink(start_time, output)
            self.cache = []
        return output

    def _process_normal(self, start_time, inputs):
        output = self.compare(inputs, self.gallery)
        self._sink(start_time, output)
        return output

    def _process_triton(self, start_time, inputs):
        if inputs:
            if self.processed:
                self.start = perf_counter()
                self.processed = False
            result = []
            self.metrics["inference"]["iters"] += 1
            self.batch_ids.append(start_time)
            for i in range(len(inputs)):
                input_crop = resize_numpy(inputs[i], self.infer_shape)
                self.cache.append(input_crop)
            if len(self.cache) < self.batch:
                return None
            else:
                l = list(self.extractor.infer([np.array(self.cache)]).values())
                result.append(l[0][0])
                output = self.compare(Tensor(result), Tensor(self.gallery))
        
                output = {'inputs': inputs, 'result': output}
                self.cache = []
                infertime = perf_counter() - self.start
                self.metrics["inference"]["total"] += infertime
                self.metrics["inference"]["list"].append(infertime)
                self.start = perf_counter()
                self._sink(start_time, output)
        else:
            self._sink(start_time, None)
    
    def _create_requests(self, start_time, inputs):
        if inputs:
            self.cache.extend([resize_numpy(i, self.infer_shape)
                               for i in inputs])
            self.batch_ids.append(start_time)
            self.metrics["inference"]["iters"] += 1
            if len(self.cache) <= self.batch:
                pass
            else:
                requests = self.extractor.prepare_requests(
                    np.array(self.cache))
                self.cache = []
                self._infer_queue.put({
                    "start_time": start_time,
                    "inputs": requests
                })
        else:
            self._sink(start_time, None)

    def _process_async(self):
        while True:
            # * Check run flag and stop if necessary
            if not self._RUN_THREAD:
                break
            request_inputs = []
            request_outputs = []
            request_starts = []
            # start = perf_counter() # ? Reenable?
            for _ in range(self.concurrency):
                request = self._infer_queue.get()
                request_inputs.append(request["inputs"][0])
                request_outputs.append(request["inputs"][1])
                request_starts.append(request["start_time"])

            start = perf_counter()
            results = self.extractor.process_async(
                request_inputs, request_outputs)
            infertime = perf_counter() - start
            self.metrics["inference"]["total"] += infertime
            self.metrics["inference"]["list"].append(infertime)
            for start, result in zip(request_starts, results):
                self._sink(start, result)

    def compare(self, tensor1, tensor2):
        return compute_distance_matrix(tensor1.to(self.device), tensor2.to(self.device), self.metric)

    def __all__(self):
        return ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0']

    def _fakesink(self, start_time, args):
        if args:
            for t in self.batch_ids:
                time_taken = time() - t
                self.metrics["pipeline"]["total"] += time_taken
                self.metrics["pipeline"]["iters"] += 1
                self.metrics["pipeline"]["list"].append(time_taken)
    
            self.batch_ids = []
        else:
            # * If we dont have args this means there wasn't anything to process.
            super()._fakesink(start_time, args)


    def _metrics_setup(self):
        super()._metrics_setup()
        self.metrics["inference"] = {
            "help": "Inference metrics for reID. Total time taken, number of iterations and per batch process times.",
            "total": 0,
            "iters": 0,
            "list": []
        }
