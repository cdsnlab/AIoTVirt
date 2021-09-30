# from gevent import monkey
# monkey.patch_all()
import tritonclient.http as httpclient
import requests
import numpy as np
import time

from line_profiler import LineProfiler
profile = LineProfiler()
import atexit
# atexit.register(profile.print_stats)

async_time = 0
result_time = 0
output_time = 0
iters = 0

def print_line_times():
    print("Infer line: ", async_time / iters)
    print("result line: ", result_time / iters)
    print("compare line: ", output_time / iters)

# atexit.register(print_line_times)

class Tritonclient():
    def __init__(self, model, infer_url, concurrency):
        self.model = model
        self.infer_url = infer_url
        # list of dictionary [(name:str, shape:list, datatype:str),..]
        self.input = None
        # list of dictionary [(name:str, shape:list, datatype:str),..]
        self.output = None
        self.triton_client = httpclient.InferenceServerClient(
            url=self.infer_url, verbose=False, concurrency=concurrency)
        self.infer_time = 0
        self.infer_num = 0

    def view_metadata(self, model, url):
        model_meta_url = 'http://' + url + '/v2/models/' + model
        response = requests.get(model_meta_url)
        if response.status_code != 200:
            print('Bad request', response.status_code)
            return None
        else:
            metadata = response.json()
            self.input = metadata['inputs']
            self.output = metadata['outputs']

    def prepare_requests(self, input_val):
        inputs = []
        outputs = []
        if self.input is None or self.output is None:
            self.view_metadata(self.model, self.infer_url)

        for index, infer_input in enumerate(self.input):
            infer_input_shape = infer_input['shape'] + []
            if input_val.shape[0] == 1:
                input_val = input_val[0]
            if infer_input['shape'][0] == -1:
                infer_input_shape[0] = input_val.shape[0]
            
            infer_input_arg = (
                infer_input['name'], infer_input_shape, infer_input['datatype'])
            input_buffer = httpclient.InferInput(*infer_input_arg)
            inputs.append(input_buffer)
            # ! Might cause problem with detection?
            input_buffer.set_data_from_numpy(
                input_val, binary_data=True)

        for index, infer_output in enumerate(self.output):
            infer_output_arg = (infer_output['name'])
            output_buffer = httpclient.InferRequestedOutput(infer_output_arg)
            outputs.append(output_buffer)

        return inputs, outputs

    def process_requests(self, inputs, outputs):
        async_request = self.triton_client.async_infer(
            model_name=self.model, inputs=inputs, outputs=outputs)
        result = async_request.get_result()

        result_dict = {}
        for infer_output in self.output:
            output_name = infer_output['name']
            result_dict[output_name] = result.as_numpy(output_name)
        return result_dict

    # @profile
    def process_async(self, request_inputs, request_outputs):
        async_requests = []
        for inputs, outputs in zip(request_inputs, request_outputs):
            async_requests.append(self.triton_client.async_infer(
                model_name=self.model, inputs=inputs, outputs=outputs))

        results = []
        for request in async_requests:
            result = request.get_result()
            result_dict = {}
            for infer_output in self.output:
                output_name = infer_output['name']
                result_dict[output_name] = result.as_numpy(output_name)

            results.append(result_dict)

        return results

    # @profile
    def infer(self, input_val):
        global async_time, result_time, output_time, iters
        start_t = time.time()

        inputs = []
        outputs = []
        if self.input is None or self.output is None:
            self.view_metadata(self.model, self.infer_url)
        for index, infer_input in enumerate(self.input):
            infer_input_shape = infer_input['shape'] + []  # copy value
            if infer_input['shape'][0] == -1:
                infer_input_shape[0] = len(input_val[0])
            infer_input_arg = (
                infer_input['name'], infer_input_shape, infer_input['datatype'])
            input_buffer = httpclient.InferInput(*infer_input_arg)
            inputs.append(input_buffer)
            input_buffer.set_data_from_numpy(
                input_val[index], binary_data=True)

        for index, infer_output in enumerate(self.output):
            infer_output_arg = (infer_output['name'])
            output_buffer = httpclient.InferRequestedOutput(infer_output_arg)
            outputs.append(output_buffer)

        # * Current code, trying to change async to sync
        # i0 = time.perf_counter()
        # async_request = self.triton_client.async_infer(
        #     model_name=self.model, inputs=inputs, outputs=outputs)
        # i1 = time.perf_counter()
        # result = async_request.get_result()
        # i2 = time.perf_counter()


        i0 = time.perf_counter()
        result = self.triton_client.infer(model_name=self.model, inputs=inputs, outputs=outputs)
        i1 = time.perf_counter()
        i2 = time.perf_counter()


        result_dict = {}
        for infer_output in self.output:
            output_name = infer_output['name']
            result_dict[output_name] = result.as_numpy(output_name)
        i3 = time.perf_counter()

        async_time += i1 - i0
        result_time += i2 - i1
        output_time += i3 - i2
        iters += 1

        self.infer_num += len(input_val[0])
        self.infer_time += time.time() - start_t

        return result_dict
