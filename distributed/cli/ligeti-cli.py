from collections import deque
import os
import sys
import tensorrt as trt
import importlib.util
import numpy as np
import grpc
import asyncio
import json
import logging
from torch.utils.data import DataLoader
from logging.config import dictConfig
import datetime
from pickle import dumps
from google.protobuf.timestamp_pb2 import Timestamp

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)

try:
    from dataset.retrain_dataset_preparer import RetrainingDatasetPreparer
    from distributed.cli.trt_inference import allocate_buffers, do_inference
    import distributed.cli.ligeti_inter_data_pb2_grpc as ligeti_grpc_server
    import distributed.cli.ligeti_inter_data_pb2 as ligeti_grpc_msg
    from models.model_split import model_top_layers, split_head
    from pytorch_onnx import pytorch2onnx_convert
    from onnx_tensorrt import onnx2trt_convert
except ModuleNotFoundError:
    raise

MSG_CODE = {
    'inter_data': 1,
    'inter_data_confirm': 2,
    'config_sync': 3,
    'config_sync_confirm': 4,
    'profile_ready': 5,
    'profile_ready_confirm': 6
}


def json_list_to_py(json_list, key_list):
    py_list = []
    for e in json_list:
        print(e)
        temp_list = []
        for key in key_list:
            temp_list.append(e[key])
        py_list.append(tuple(temp_list))
    return py_list


class LigetiClient():
    def __init__(
        self,
        data_config_path='/home/ligeti/distributed/cli/default_config.py',
        server_ip: str = '143.248.55.76',
        server_port: str = '5001'
    ):
        # Loading the config from a file allows flexible argument parsing
        self.config = self.import_from_path(data_config_path).config()

        # The should be two folder inside the "logs" folder
        # One is `client`, the other `server`
        self.log_folder_path = os.path.join(BASE_DIR, 'logs', 'client')

        now = datetime.datetime.now()
        self.config.time_stamp = now.strftime('%Y-%m-%dT%H:%M:%S') + \
            ('-%02d' % (now.microsecond / 10000))
        self.run_name = '{}_{}_batch_{}_{}'.format(
            self.config.model_name,
            self.config.dataset_name,
            self.config.batch_size,
            self.config.time_stamp
        )
        self.run_root = os.path.join(self.log_folder_path, self.run_name)
        os.makedirs(self.run_root)

        # Each config file should have a path to a log config file, which
        # regulates the format of the logs
        with open(self.config.log_config_path, 'r') as log_config_file:
            self.log_config = json.load(log_config_file)

        for handler in self.log_config['handlers']:
            try:
                self.log_config['handlers'][handler]['filename'] = \
                    os.path.join(
                        os.path.join(self.run_root),
                        self.log_config['handlers'][handler]['filename'],
                    )
            except KeyError:
                pass
        if self.config.dev:
            self.logger_name = 'dev_logger'
        else:
            self.logger_name = 'prod_logger'
        dictConfig(self.log_config)
        self.logger = logging.getLogger(self.logger_name)
        self.logger.info('Finished preparing the logging process.')
        self.logger.info('Start logging.')

        input()
        self.num_retrain_tasks = len(self.config.task_specifications)
        self.model_name = self.config.model_name
        self.dataset_name = self.config.dataset_name
        self.split_point_list = self.config.split_point_list
        self.batch_size = self.config.batch_size
        self.num_classes = self.config.num_classes_for_pretrain
        self.lr = self.config.learning_rate
        self.img_num_channels = 3
        self.img_height = self.config.img_height
        self.img_width = self.config.img_width
        self.fp = self.config.fp
        with open(self.config.interdata_shape_dict_path, 'r') as f:
            try:
                self.interdata_shape_dict = json.load(f)
                f.close()
            except json.decoder.JSONDecodeError:
                self.interdata_shape_dict = {}
        self.server_ip = server_ip
        self.server_port = server_port
        self.inbound_queue = deque()
        self.outbound_queue = deque()

        # for task_num in range(self.num_retrain_tasks):
        #     self.profile(task_num)

    def log_out(self):
        with open(self.config.interdata_shape_dict_path, 'w') as f:
            json.dump(self.interdata_shape_dict, f)
            f.close()

    def grpc_server_on(self, channel, timeout=3) -> bool:
        """grpc_server_on Verify if the connection is successful

        Parameters
        ----------
        channel : grpc.aio.Channel
            An GRPC channel object

        Returns
        -------
        bool
            Whether the channel is on
        """
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            return True
        except grpc.FutureTimeoutError:
            return False

    async def main(self):
        try:
            channel_options = [
                ('grpc.max_send_message_length', 1024 * 1024 * 1024),
                ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
            ]

            self.channel = grpc.insecure_channel('{}:{}'.format(
                self.server_ip,
                self.server_port
                ), options=channel_options
            )
            if self.grpc_server_on(self.channel):
                self.logger.info('Successfully connected to {}:{}'.format(
                    self.server_ip,
                    self.server_port
                ))
            else:
                self.logger.debug('Failed to connect to the server.')

            self.stub = ligeti_grpc_server.LigetiProfileStub(self.channel)
            loop = asyncio.get_event_loop()

            # Running two concurrent tasks
            # One (`send`) constantly checks if there is anything put into
            # the `outbound queue` and send them to server.
            # The other runs the profiling for the current retraining task.
            tasks = [
                loop.create_task(self.send()),
                loop.create_task(self.profile(task_num=0))
            ]
            await asyncio.gather(*tasks)
        finally:
            self.log_out()

    def config_sync(self, split_point):
        config_msg = ligeti_grpc_msg.ConfigSend(
            msg_type=MSG_CODE['config_sync'],
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            split_point=split_point,
            num_batches=1000//self.batch_size,
            lr=self.lr
        )
        resp = self.stub.config_sync(config_msg, 1)
        return resp.ack

    def profile_ready_signal(self):
        current_time = Timestamp()
        current_time.GetCurrentTime()
        msg = ligeti_grpc_msg.ProfileReadySignal(
            msg_type=MSG_CODE['profile_ready'],
            timestamp=current_time
        )

        resp = self.stub.profile_ready_signal(msg, 1)
        return resp

    async def send(self):
        while True:
            try:
                # print('send')
                nxt_outbound_data = self.outbound_queue.popleft()
                current_time = Timestamp()
                current_time.GetCurrentTime()
                out_data_shape = ligeti_grpc_msg.DataShape(
                    num_channels=nxt_outbound_data[3]['num_channels'],
                    height=nxt_outbound_data[3]['height'],
                    width=nxt_outbound_data[3]['width']
                )
                out_msg = ligeti_grpc_msg.InterData(
                    msg_type=MSG_CODE['inter_data'],
                    split_point=nxt_outbound_data[0],
                    batch_num=nxt_outbound_data[1],
                    inter_data=nxt_outbound_data[2],
                    data_shape=out_data_shape,
                    timestamp=current_time
                )
                resp = self.stub.inter_data_stream(out_msg, 1)
                # print(resp)
                # return resp
            except IndexError:
                pass

            await asyncio.sleep(1/1000)

    async def profile(self, task_num):
        retrain_dataset = RetrainingDatasetPreparer(
            self.config.dataset_name,
            self.config.data_dir_path,
            self.config.num_classes_for_pretrain,
            self.config.num_imgs_from_chosen_pretrain_classes,
            self.config.num_imgs_from_chosen_test_classes,
            self.config.choosing_class_seed,
            self.config.pretrain_train_data_shuffle_seed,
            self.config.pretrain_test_data_shuffle_seed,
            self.config.task_specifications[0],
            task_num,
            self.config.retrain_data_shuffle_seed,
            transforms=self.config.transforms
        )
        retrain_dataloader = DataLoader(
            dataset=retrain_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=True
        )
        for imgs, classes in retrain_dataloader:
            print(imgs.shape, classes.shape)
        input()
        data = np.random.rand(self.batch_size, 3, 32, 32).astype(np.float32)
        # self.convert_model()
        print(1)
        data = np.ones((3, 32, 32))
        print(data.shape)
        for split_point in self.split_point_list:
            if split_point == 0:
                continue
            print(split_point)
            self.config_sync(split_point)
            trt_model_name = \
                '{}_{}_split_{}_shape_{}_{}_{}_batch_{}_fp{}'.format(
                    self.model_name,
                    self.dataset_name,
                    split_point,
                    self.img_num_channels,
                    self.img_height,
                    self.img_width,
                    self.batch_size,
                    self.fp
                )
            trt_model_path = os.path.join(
                BASE_DIR,
                'distributed/cli/trt_models',
                trt_model_name+'.trt'
            )
            print(trt_model_name)
            output_shape = \
                self.interdata_shape_dict[trt_model_name]
            print(output_shape)
            self.load_trt_model(trt_model_path)
            for batch_num in range(30):
                outputs = do_inference(
                    context=self.context,
                    bindings=self.bindings,
                    inputs=self.inputs,
                    outputs=self.outputs,
                    stream=self.stream,
                    batch_size=1
                )
                # final_outputs = np.reshape(outputs[0], (-1, 64, 16, 16))
                # print(final_outputs)
                print(outputs[0].shape)
                serialized_out_data = dumps(outputs[0])
                self.outbound_queue.append((
                    split_point,
                    batch_num,
                    serialized_out_data,
                    output_shape
                ))
                await asyncio.sleep(1/1000)
            break

    def load_trt_model(self, engine_path):
        runtime = trt.Runtime(TRT_LOGGER)
        assert runtime

        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        self.context = engine.create_execution_context()
        assert self.context

        self.inputs, self.outputs, self.bindings, self.stream = \
            allocate_buffers(engine)

    def import_from_path(self, path):
        spec = importlib.util.spec_from_file_location(
            'config', path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['config'] = module
        spec.loader.exec_module(module)
        return module

    def convert_model(self):
        top_layer = model_top_layers[self.model_name]
        for layer_num in range(top_layer):
            if layer_num == 0:
                continue
            head_model = split_head(
                model_name=self.model_name,
                split_point=layer_num
            )
            onnx_model_name = self.model_name + \
                '_{}_split_{}_shape_{}_{}_{}'.format(
                    self.dataset_name,
                    layer_num,
                    3,
                    self.img_height,
                    self.img_width
                )
            onnx_model = pytorch2onnx_convert(
                head_model,
                os.path.join(BASE_DIR, 'distributed/cli/onnx_models'),
                task='classification',
                name=onnx_model_name,
                batch_size=self.batch_size,
                num_channels=3,
                height=self.img_height,
                width=self.img_width
            )
            _, trt_model_name, output_shape = onnx2trt_convert(
                onnx_model,
                os.path.join(BASE_DIR, 'distributed/cli/trt_models'),
                fp=self.fp,
                num_channels=3,
                height=self.img_height,
                width=self.img_width
            )
            self.interdata_shape_dict[trt_model_name] = {
                'num_channels': output_shape[1],
                'height': output_shape[2],
                'width': output_shape[3]
            }
            break


if __name__ == '__main__':
    ligeti_client = LigetiClient()
    asyncio.get_event_loop().run_until_complete(ligeti_client.main())
