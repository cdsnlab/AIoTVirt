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
    fp16=False,
    verbose=False
):

    onnx_model_name = onnx_model_path.split('/')[-1].split('.')[0]
    trt_model_name = onnx_model_name + '_' + str(batch_size) + '.trt'
    trt_model_path = os.path.join(trt_folder_path, trt_model_name)
    TRT_LOGGER = trt.Logger()
    explicit_batch = 1 << \
        (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = batch_size
        builder.max_workspace_size = 2 << 30
        print('Loading ONNX file from path {}...'.format(onnx_model_name))
        with open(onnx_model_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
        if parser.num_errors == 0:
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.
                  format(onnx_model_name))

            if fp16:
                builder.fp16_mode = True

            network.get_input(0).shape = [batch_size, 3, 224, 224]
            print(network.get_output(0).name, network.get_output(0).shape)
            engine = builder.build_cuda_engine(network)
            print("engine:",engine)
            if engine is not None:
                print("Completed creating Engine")
                with open(trt_model_path, "wb") as f:
                    f.write(engine.serialize())
                return engine
        else:
            print('Number of errors: {}'.format(parser.num_errors))
            for i in range(parser.num_errors):
                # if it gets mnore than one error this have to be changed
                error = parser.get_error(i)
                desc = error.desc()
                line = error.line()
                code = error.code()
                print('Description of the error: {}'.format(desc))
                print('Line where the error occurred: {}'.format(line))
                print('Error code: {}'.format(code))
            print("Model was not parsed successfully")


if __name__ == '__main__':
    onnx2trt_convert(
        onnx_model_path='/home/ligeti/distributed/cli/onnx_models/resnet18.onnx',
        trt_folder_path='/home/ligeti/distributed/cli/trt_models'
    )