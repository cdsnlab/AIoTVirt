import os
import sys
import tensorrt as trt

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(BASE_DIR)


def onnx2trt_convert(
    onnx_model_path: str,
    trt_folder_path: str,
    batch_size=32,
    num_channels=3,
    height=224,
    width=224,
    fp=32,
    inference_ready=False,
    logger=None
):
    def log(msg):
        if logger is None:
            return
        logger.debug(msg)
    onnx_model_name = onnx_model_path.split('/')[-1].split('.')[0]
    trt_model_name = '{model_name}_batch_{batch_size}_fp{fp}'.format(
        model_name=onnx_model_name,
        batch_size=batch_size,
        fp=fp
    )
    trt_model_path = os.path.join(trt_folder_path, trt_model_name + '.trt')
    TRT_LOGGER = trt.Logger()
    explicit_batch = 1 << \
        (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size
        builder.max_workspace_size = 2 << 30
        log('Loading ONNX file from path {}...'
            .format(onnx_model_path))
        with open(onnx_model_path, 'rb') as model:
            log('Beginning ONNX file parsing.')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        if parser.num_errors == 0:
            log('Completed parsing of ONNX file')
            log('Building an engine from file; this may take a while...')

            if fp == 16:
                builder.fp16_mode = True
                log('Using fp{}'.format(fp))

            network.get_input(0).shape = [
                batch_size,
                num_channels,
                height,
                width
            ]
            log('The shape of input is {}'.format((
                batch_size,
                num_channels,
                height,
                width
            )))
            log('The shape of output is {}'.format(
                network.get_output(0).shape
            ))
            output_shape = network.get_output(0).shape
            engine = builder.build_cuda_engine(network)
            log('engine: {}'.format(engine))
            if engine is not None:
                log("Completed creating Engine")
                with open(trt_model_path, "wb") as f:
                    log('File written to {}'.format(trt_model_path))
                    f.write(engine.serialize())
                    f.close()
                if inference_ready:
                    return engine, trt_model_name, output_shape
                else:
                    return trt_model_path, trt_model_name, output_shape
        else:
            log('Number of errors: {}'.format(parser.num_errors))
            for i in range(parser.num_errors):
                # if it gets mnore than one error this have to be changed
                error = parser.get_error(i)
                desc = error.desc()
                line = error.line()
                code = error.code()
                log('Description of the error: {}'.format(desc))
                log('Line where the error occurred: {}'.format(line))
                log('Error code: {}'.format(code))
            log("Model was not parsed successfully")


if __name__ == '__main__':
    onnx2trt_convert(
        '/home/ligeti/distributed/cli/onnx_models/resnet18.onnx',
        '/home/ligeti/distributed/cli/trt_models'
    )
