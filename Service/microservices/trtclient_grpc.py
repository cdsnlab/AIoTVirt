import tritonclient.grpc as grpcclient
import time

class Tritonclient():
    def __init__(self, model, infer_url):
        self.model = model
        self.infer_url = infer_url
        # list of dictionary [(name:str, shape:list, datatype:str),..]
        self.input = None
        # list of dictionary [(name:str, shape:list, datatype:str),..]
        self.output = None
        self.triton_client = grpcclient.InferenceServerClient(
            url=self.infer_url.replace("8000", "8001"), verbose=False)
        self.infer_time = 0
        self.infer_num = 0

    def view_metadata(self, model, url):
        model_metadata = self.triton_client.get_model_metadata(model_name=model)

        input_metadata = list(model_metadata.inputs)
        output_metadata = list(model_metadata.outputs)
        self.input = input_metadata
        self.output = output_metadata

    def prepare_requests(self, input_val):
        inputs = []
        outputs = []
        if self.input is None or self.output is None:
            self.view_metadata(self.model, self.infer_url)

        for infer_input in self.input:
            infer_input_shape = list(infer_input.shape) + []
            if input_val.shape[0] == 1:
                input_val = input_val[0]
            if infer_input.shape[0] == -1:
                infer_input_shape[0] = input_val.shape[0]
            
            infer_input_arg = (
                infer_input.name, infer_input_shape, infer_input.datatype)
            input_buffer = grpcclient.InferInput(*infer_input_arg)
            inputs.append(input_buffer)
            
            input_buffer.set_data_from_numpy(input_val)

        for infer_output in self.output:
            outputs.append(grpcclient.InferRequestedOutput(infer_output.name))

        return inputs, outputs

    # TODO This doesnt work as grpc uses callback for async
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
                result_dict[infer_output.name] = result.as_numpy(infer_output.name)

            results.append(result_dict)

        return results

    def infer(self, input_val):
        global async_time, result_time, output_time, iters
        start_t = time.time()

        inputs = []
        outputs = []
        if self.input is None or self.output is None:
            self.view_metadata(self.model, self.infer_url)

        # * Create Inputs
        for index, infer_input in enumerate(self.input):
            infer_input_shape = list(infer_input.shape) + []  # copy value
            if infer_input_shape[0] == -1:
                infer_input_shape[0] = len(input_val[0])
            infer_input_arg = (
                infer_input.name, infer_input_shape, infer_input.datatype)
            input_buffer = grpcclient.InferInput(*infer_input_arg)
            inputs.append(input_buffer)
            input_buffer.set_data_from_numpy(input_val[index])

        # * Create Outputs
        for index, infer_output in enumerate(self.output):
            infer_output_arg = (infer_output.name)
            output_buffer = grpcclient.InferRequestedOutput(infer_output_arg)
            outputs.append(output_buffer)

        # * Infer
        result = self.triton_client.infer(model_name=self.model, inputs=inputs, outputs=outputs)

        # * Convert results to numpy
        result_dict = {}
        for infer_output in self.output:
            result_dict[infer_output.name] = result.as_numpy(infer_output.name)

        self.infer_num += len(input_val[0])
        self.infer_time += time.time() - start_t

        return result_dict
