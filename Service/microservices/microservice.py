import asyncio
import http.client
from time import perf_counter, time, sleep
import threading
from queue import Queue
import msgpack
import msgpack_numpy
import aiohttp

msgpack_numpy.patch()


class Microservice(object):
    def __init__(self, name: str = None, is_sink: bool = False, is_local: bool = True, url: str = None,
                 port: int = None, is_head: bool = False, outputs: list = []):
        """[summary]

        Args:
            name (str, optional): Name of microservice. Defaults to microservice class name.
            is_sink (bool, optional): Is the microservice a sink element. Defaults to False.
            is_local (bool, optional): Is the microservice a local element. Defaults to True.
            is_head (bool, optional): Is the microservice a head element. Defaults to False.
        """
        self.links = list()
        if name:
            self.name = name
        else:
            self.name = self.__str__()
        self.is_sink = is_sink
        self.is_local = is_local
        self.is_head = is_head
        self.outputs = outputs  # List of string name Microservice outputs
        self._metrics_setup() # * Setup metrics dict
        self.queues = {}
        # ! Temporary maxsize to avoid memory related errors
        self.queue = self._setup_queue("main", 100, "Queue of main thread of service")
        self._threads = [] # * List of references to all threads so they can be terminated
        self._RUN_THREAD = True
        self.thread = self._spawn_thread(
            target=self._thread, args=(), daemon=True)
        if not self.is_head:
            self.thread.start()

        if self.is_sink:
            self._sink = self._fakesink
        else:
            self._sink = self._sink

        if self.is_local:
            self.src = self._local_src
        else:
            self.src = self._async_remote
            self.url = url
            self.port = port
            self.req_queue  = self._setup_queue("request_queue", 100)
            self.data_queue = self._setup_queue("data_queue", 100)
            self.data_thread = self._spawn_thread(
                target=self._pack, args=(), daemon=True)
            self.req_thread = self._spawn_thread(
                target=self._as_thread, args=(), daemon=True)
            self.data_thread.start()
            self.req_thread.start()

    def start(self):
        # Ran only once if head of pipeline
        if not self.is_head:
            print("Can't use this method if Microservice isn't head of pipeline!")
        else:
            pass

    def _thread(self):
        """[summary]
        Microservice processing thread. Times execution of each iteration
        Shouldn't need to be overwritten
        """
        while True:
            # * Check run flag and stop if necessary
            if not self._RUN_THREAD:
                break
            inputs = self.queue.get()
            start = perf_counter()
            self._process(**inputs)
            processTime = perf_counter() - start
            self.metrics["process"]["total"] += processTime
            self.metrics["process"]["list"].append(processTime)
            self.metrics["process"]["iters"] += 1

    def cleanup(self):
        self._kill_threads()
        sleep(2)
        del self.metrics
        del self.queues

    def _spawn_thread(self, target=None, args=(), daemon=True):
        """[summary]
        Wrapper to spawn new threads and add them to the list of threads
        """
        thread = threading.Thread(
            target=target, args=args, daemon=daemon)
        self._threads.append(thread)
        return thread

    def _process(self, input):
        pass

    def _kill_threads(self, timeout=2):
        """[summary]
        Stop the processing thread

        Args:
            timeout (int, optional): Timeout to wait for joining thread. Defaults to 2.
        """
        self._RUN_THREAD = False
        for t in self._threads:
            if t.is_alive():
                t.join(timeout)
                print("Killed thread of service: ", self.name)
            else:
                print("Thread not running")

    def link(self, link):
        if type(link) == list:
            self.links.extend(link)
        else:
            self.links.append(link)

    def prepare(self, inputs):
        """[summary]
        Called only when from remote. Web server passes decoded object.
        Stored in processing queue
        Args:
            inputs (object): Input object
        """
        self.queue.put({
            "start_time": inputs["start_time"],
            "inputs": inputs["inputs"]
        })

    def _local_src(self, start_time, inputs):
        self.queue.put({
            "start_time": start_time,
            "inputs": inputs
        })

    def _pack(self):
        while True:
            # * Check run flag and stop if necessary
            if not self._RUN_THREAD:
                break
            data = self.data_queue.get()
            data = msgpack.packb(data)
            self.req_queue.put(data)

    async def _make_requests(self):
        async with aiohttp.ClientSession() as session:
            while True:
                # * Check run flag and stop if necessary
                if not self._RUN_THREAD:
                    break
                data = self.req_queue.get()
                p2 = perf_counter()
                # * Header to log how long it takes to receive the request
                headers = { 'send-time': str(time())}
                async with session.post("http://{}:{}/remote_src".format(self.url, self.port), data=data, headers=headers) as resp:
                    pass
                self.metrics["remote_request"]["list"].append(perf_counter() - p2)

    def _as_thread(self):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._make_requests())
        loop.close


    def _async_remote(self, start_time, inputs):
        data = {
            "service": self.name,
            "inputs": {
                "start_time": start_time,
                "inputs": inputs
            }
        }
        self.data_queue.put(data)

    def _sink(self, start_time, args):
        """[summary]
        Pass output from this Microservice to all connected components
        Args:
            args ([object]): Output object(s)
        """
        for link in self.links:
            link.src(start_time, args)

    def _fakesink(self, start_time, args):
        """[summary]
        Fakesink that does nothing
        Args:
            args ([object]): Object placeholder. Takes any num of args
        """
        self.metrics["pipeline"]["list"].append(time() - start_time)
        self.metrics["pipeline"]["total"] += time() - start_time
        self.metrics["pipeline"]["iters"] += 1

    def _metrics_setup(self):
        self.metrics = {
            "process": {
                "help": "Metrics of _process function in main thread of microservice. Total time and iterations",
                "total": 0,
                "iters": 0,
                "list": []
            }
        }

        self.metrics["network_time"] = {
            "list": []
        }

        if self.is_sink:
            self.metrics["pipeline"]  = {
                "help": "Total end-to-end latency time",
                "total": 0,
                "iters": 0,
                "list": []
            }

        if not self.is_local:
            self.metrics["remote_request"] = {
                "help": "List of times to send request to next component in line",
                "list": []
            }

    def get_metrics(self):
        response = {}
        for metric, values in self.metrics.items():
            response[metric] = values.copy()
            if "list" in self.metrics[metric]:
                self.metrics[metric]["list"] = []

        return response

    def _setup_queue(self, name: str, maxsize: int = 100, help: str = None):
        """[summary]
        Wrapper to create new queues and add them to the dict of queues

        Args:
            name (str): Name identifier of queue
            maxsize (int, optional): Maximum size of queue. Default is 100
            help (str, optional): Description string of queue
        """
        queue = Queue(maxsize)
        self.queues[name] = {
            "help": help,
            "queue": queue
        }
        return queue

    def add_network_time(self, network_time):
        self.metrics["network_time"]["list"].append(network_time)

    def get_queue_capacity(self):
        capacity = {}
        for name, value in self.queues.items():
            capacity[name] = {
                "help": value["help"],
                "size": value["queue"].qsize()
            }

        return capacity

    def __str__(self) -> str:
        return self.__class__.__name__
