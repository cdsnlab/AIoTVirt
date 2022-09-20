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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)

try:
    from dataset.retrain_dataset_preparer import RetrainingDatasetPreparer,\
        collate_fn
    from distributed.cli.trt_inference import do_inference, load_trt_model
    import distributed.cli.ligeti_inter_data_pb2_grpc as ligeti_grpc_server
    import distributed.cli.ligeti_inter_data_pb2 as ligeti_grpc_msg
    from models.model_split import model_top_layers, split_head
    from pytorch_onnx import pytorch2onnx_convert
    from onnx_tensorrt import onnx2trt_convert
    from utils import import_from_path
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
        data_config_path='/home/ligeti/distributed/cli/default_config.py'
    ):
        # Loading the config from a file allows flexible argument parsing
        self.config = import_from_path(data_config_path).config()

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
        self.log_config_path = os.path.join(
            BASE_DIR,
            self.config.log_config_path
        )
        with open(self.log_config_path, 'r') as log_config_file:
            self.log_config = json.load(log_config_file)

        for handler in self.log_config['handlers']:
            try:
                self.log_config['handlers'][handler]['filename'] = \
                    os.path.join(
                        self.run_root,
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

        self.num_retrain_tasks = len(self.config.task_specifications)
        self.model_name = self.config.model_name
        self.dataset_name = self.config.dataset_name
        self.batch_size = self.config.batch_size
        self.num_classes = self.config.num_classes_for_pretrain
        self.logger.info('There are {} retraining task(s) for model'
                         '{} with {} class(es) using dataset {} with'
                         'a batch size of {}'.format(
                            self.num_retrain_tasks,
                            self.model_name,
                            self.num_classes,
                            self.dataset_name,
                            self.batch_size
                         ))
        self.split_point_list = self.config.split_point_list
        self.logger.info('Potential split points {}'.format(
            self.split_point_list
        ))
        self.lr = self.config.learning_rate
        self.img_num_channels = 3
        self.img_height = self.config.img_height
        self.img_width = self.config.img_width
        self.fp = self.config.fp
        self.logger.info('Client side uses fp{}'.format(self.fp))

        with open(self.config.interdata_shape_dict_path, 'r') as f:
            try:
                self.interdata_shape_dict = json.load(f)
                f.close()
            except json.decoder.JSONDecodeError:
                self.interdata_shape_dict = {}
        self.logger.info('Load dictionary for intermediate data\'s shape')
        self.inbound_queue = deque()
        self.outbound_queue = deque()
        # for task_num in range(self.num_retrain_tasks):
        #     self.profile(task_num)

    def save_shape_dict(self):
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
                self.config.server_ip,
                self.config.server_port
                ), options=channel_options
            )
            if self.grpc_server_on(self.channel):
                self.logger.info('Successfully connected to {}:{}'.format(
                    self.config.server_ip,
                    self.config.server_port
                ))
            else:
                self.logger.debug('Failed to connect to the server, at {}:{}.'
                                  .format(
                                      self.config.server_ip,
                                      self.config.server_port
                                  ))

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
            self.save_shape_dict()

    def config_sync(self, split_point):
        config_msg = ligeti_grpc_msg.ConfigSend(
            msg_type=MSG_CODE['config_sync'],
            model_name=self.model_name,
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            split_point=split_point,
            num_batches=self.num_batches,
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
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=collate_fn
        )
        self.num_batches = len(retrain_dataloader)
        # self.convert_model()
        self.save_shape_dict()
        self.logger.info('Saved inter data shapes at {}'
                         .format(
                             self.config.interdata_shape_dict_path
                         ))
        self.logger.info('Start profiling for model {} with dataset {} '
                         'at different split points.'.format(
                            self.model_name,
                            self.dataset_name
                         ))
        for split_point in self.split_point_list:
            if split_point == 0:
                continue
            self.logger.info('Proling at split point {}'.format(
                split_point
            ))

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
            output_shape = \
                self.interdata_shape_dict[trt_model_name]
            inputs, outputs, bindings, stream, context = \
                load_trt_model(trt_model_path)
            self.logger.info('Loaded tensorrt model at {}'.format(
                trt_model_path
            ))
            self.logger.debug('Output of {} is {}'.format(
                trt_model_name,
                output_shape
            ))
            resp = self.profile_ready_signal()
            if resp is None:
                self.logger.info('Server failed to acknowledge that client '
                                 'is ready.')
            else:
                self.logger.info('Server acknowledged that client is ready. '
                                 'Proceed to send data')
            for batch_num, (imgs, classes) in enumerate(retrain_dataloader):
                inputs[0].host = imgs
                trt_outputs = do_inference(
                    context=context,
                    bindings=bindings,
                    inputs=inputs,
                    outputs=outputs,
                    stream=stream,
                    batch_size=self.batch_size
                )
                print(trt_outputs[0].shape)
                # serialized_out_data = dumps(outputs[0])
                # self.outbound_queue.append((
                #     split_point,
                #     batch_num,
                #     serialized_out_data,
                #     output_shape
                # ))
                # await asyncio.sleep(1/1000)
            break

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
        self.logger.info('Converting partial models of {} with dataset {} '
                         'at different split points'.format(
                            self.model_name,
                            self.dataset_name
                         ))
        for layer_num in range(top_layer):
            if layer_num == 0:
                continue
            self.logger.info('Converting at split layer {}'
                             .format(layer_num))
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
            self.logger.debug('Successfully converted pytorch to onnx model.'
                              'Stored at {}'.format(
                                onnx_model
                              ))
            trt_model_path, trt_model_name, output_shape = onnx2trt_convert(
                onnx_model,
                os.path.join(BASE_DIR, 'distributed/cli/trt_models'),
                fp=self.fp,
                batch_size=self.batch_size,
                num_channels=3,
                height=self.img_height,
                width=self.img_width,
                logger=self.logger
            )
            self.logger.debug('Successfully converted onnx to tensorrt model.'
                              'Stored at {}'.format(
                                trt_model_path
                              ))
            self.logger.debug('Output shape of model {} is {}'.format(
                trt_model_name,
                output_shape
            ))
            self.interdata_shape_dict[trt_model_name] = {
                'num_channels': output_shape[1],
                'height': output_shape[2],
                'width': output_shape[3]
            }
        self.logger.info('Successfully converted partial models of {} with' 
                         'dataset {} at ALL split points'.format(
                            self.model_name,
                            self.dataset_name
                         ))


if __name__ == '__main__':
    ligeti_client = LigetiClient()
    asyncio.get_event_loop().run_until_complete(ligeti_client.main())
