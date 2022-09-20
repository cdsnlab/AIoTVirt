import os
import sys
import grpc
import asyncio
import numpy as np
import json
from collections import deque
import datetime
import torch
import torch.nn.functional as F
import logging
from logging.config import dictConfig

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)

try:
    from models.model_split import TailModel
    from distributed.server.ligeti_grpc_server_servicer import \
        LigetiGrpcServicer, MSG_CODE
    import ligeti_inter_data_pb2_grpc as ligeti_grpc_server
    from utils import import_from_path
except ModuleNotFoundError:
    raise


class LigetiServer():
    def __init__(
        self,
        config_path='/home/ligeti/distributed/server/default_config.py'
    ):
        # Loading the config from a file allows flexible argument parsing
        self.config = import_from_path(config_path).config()

        # The should be two folder inside the "logs" folder
        # One is `client`, the other `server`
        self.log_folder_path = os.path.join(BASE_DIR, 'logs', 'server')

        now = datetime.datetime.now()
        time_stamp = now.strftime('%Y-%m-%dT%H:%M:%S') + \
            ('-%02d' % (now.microsecond / 10000))
        self.run_name = 'server_log_{}'.format(time_stamp)
        self.run_root = os.path.join(
            self.log_folder_path,
            self.run_name
        )
        os.makedirs(self.run_root)

        # Each config file should have a path to a log config file, which
        # regulates the format of the logs
        log_config_path = os.path.join(
            BASE_DIR,
            self.config.log_config_path
        )
        with open(log_config_path, 'r') as log_config_file:
            log_config = json.load(log_config_file)

        for handler in log_config['handlers']:
            try:
                log_config['handlers'][handler]['filename'] = \
                    os.path.join(
                        self.run_root,
                        log_config['handlers'][handler]['filename'],
                    )
            except KeyError:
                pass

        if self.config.dev:
            logger_name = 'dev_logger'
        else:
            logger_name = 'prod_logger'
        dictConfig(log_config)
        self.server_logger = logging.getLogger(logger_name)

        self.server_logger.info('Finished preparing the logging process.')
        self.server_logger.info('Start logging.')

        self.device = self.config.device

        self.task_logger = None
        self.inter_data_list = {}
        self.config_done = False
        self.client_model_load_done = False
        self.config_list = deque()

    def ready_to_profile(self):
        return self.client_model_load_done and self.config_done

    async def define_grpc_server(self):
        channel_options = [
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
        ]

        self.grpc_server = grpc.aio.server(options=channel_options)
        self.ligeti_grpc_servicer = LigetiGrpcServicer()
        ligeti_grpc_server.add_LigetiProfileServicer_to_server(
            self.ligeti_grpc_servicer,
            self.grpc_server
        )
        self.grpc_server.add_insecure_port('[::]:{}'.format(self.config.port))
        self.server_logger.info('Start LISTENing to port {}.'
                                .format(self.config.port))

        await self.grpc_server.start()
        await self.grpc_server.wait_for_termination()

    def config_sync(self, config):
        self.tail_model = TailModel(
            model_name=config['model_name'],
            split_point=config['split_point']
        ).to(self.device)
        self.tail_optimizer = torch.optim.SGD(
            self.tail_model.parameters(),
            lr=config['lr'],
            momentum=0.9,
            weight_decay=5e-4
        )
        self.batch_size = config['batch_size']
        self.num_classes = config['num_classes']
        self.num_batches = config['num_batches']
        self.split_point = config['split_point']
        self.data_shape = config['data_shape']
        self.temperature = config['temperature']
        self.alpha = config['alpha']
        self.num_profile_epochs = config['num_profile_epochs']
        self.num_retrain_epochs = config['num_retrain_epochs']
        self.retrain_time_budget = config['retrain_time_budget']
        self.kd_loss_function = torch.nn.KLDivLoss()
        self.server_logger.info('Set config {}'.format(config))

        if self.task_logger is None:
            task_log_config_path = os.path.join(
                BASE_DIR,
                self.config.task_log_config_path
            )

            with open(task_log_config_path, 'r') as log_config_file:
                log_config = json.load(log_config_file)

            now = datetime.datetime.now()
            time_stamp = now.strftime('%Y-%m-%dT%H:%M:%S') + \
                ('-%02d' % (now.microsecond / 10000))
            log_config['handlers']['task_file_handler']['filename'] = \
                os.path.join(
                    self.run_root,
                    'retrain_{}'.format(time_stamp)
                )
            dictConfig(log_config)
            self.task_logger = logging.getLogger('task_logger')

            self.task_logger.info('Start task logging.')

        self.config_done = True

    async def profile(self):
        while True:
            if self.ready_to_profile():
                try:
                    current_config = self.config_list.popleft()
                    self.config_sync(current_config)
                    self.config_done = True
                    batch_idx = 0
                    while batch_idx < self.num_batches:
                        try:
                            inter_data = self.inter_data_list[
                                self.split_point
                            ][batch_idx]
                            print(self.split_point, batch_idx, inter_data.shape)
                            batch_idx += 1
                        except (IndexError, KeyError):
                            print('Wait for intermediate data to come.')
                            await asyncio.sleep(1/10)
                    self.client_model_convert_done = False
                    self.config_done = False
                except IndexError:
                    await asyncio.sleep(1/10)
            else:
                await asyncio.sleep(1/10)

    async def run_server(self):
        while True:
            try:
                nxt = self.ligeti_grpc_servicer.inbound_queue.popleft()
                if nxt['msg_type'] == MSG_CODE['config_sync']:
                    self.config_list.append(nxt)
                elif nxt['msg_type'] == MSG_CODE['inter_data']:
                    inter_data = np.reshape(
                        nxt['inter_data'],
                        nxt['data_shape']
                    )
                    try:
                        self.inter_data_list[nxt['split_point']].append(
                            inter_data
                        )
                    except KeyError:
                        print('See data from a new split point')
                        self.inter_data_list[nxt['split_point']] = []
                        self.inter_data_list[nxt['split_point']].append(
                            inter_data
                        )
                elif nxt['msg_type'] == MSG_CODE['profile_ready']:
                    self.client_model_convert_done = True
            except IndexError:
                pass
            await asyncio.sleep(1/1000)

    async def main(self):

        tasks = [
            asyncio.create_task(self.define_grpc_server()),
            asyncio.create_task(self.run_server()),
            asyncio.create_task(self.profile())
        ]

        await asyncio.gather(*tasks)


if __name__ == '__main__':
    ligeti_server = LigetiServer()
    asyncio.get_event_loop().run_until_complete(ligeti_server.main())
