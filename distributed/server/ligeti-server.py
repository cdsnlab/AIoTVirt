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

    async def train_one_epoch(
        self,
        epoch,
        profile=True,
        test=True
    ):
        """run_one_epoch trains model for one single epoch

        Parameters
        ----------
        epoch : int
            The current epoch index
        """
        self.tail_model.train()
        stage = 'Profiling' if profile else 'Retraining'
        self.task_logger.info('{} split point {} epoch {}.'.format(
            stage,
            self.split_point,
            epoch
        ))
        correct = 0
        total = 0
        for batch_idx in range(self.num_batches):
            self.tail_optimizer.zero_grad()

            inter_data, targets = self.inter_data_list[self.split_point][
                batch_idx
            ]
            inter_data = torch.from_numpy(inter_data).to(self.device)
            targets = torch.from_numpy(targets).to(self.device)

            outputs = self.tail_model(inter_data)

            KD_loss = self.kd_loss_function(
                F.log_softmax(outputs/self.temperature, dim=1),
                F.softmax(
                    self.old_outputs[batch_idx]/self.temperature,
                    dim=1
                )
            ) * (self.temperature ** 2)
            CE_loss = F.cross_entropy(outputs, targets) * \
                (1. - self.alpha)
            loss = KD_loss + CE_loss
            self.task_logger.info('{} split point {} epoch {}, batch {}, '
                                  'training loss = {}'
                                  .format(
                                    stage,
                                    self.split_point,
                                    epoch,
                                    batch_idx,
                                    loss.item()
                                    ))

            loss.backward()
            self.tail_optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            await asyncio.sleep(1/1000)

        self.server_logger.info('{} split point {} epoch {}, training '
                                'accuracy {}'.format(
                                    stage,
                                    self.split_point,
                                    epoch,
                                    correct/total
                                ))

        if not test:
            return
        self.tail_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data in self.test_data:
                inter_data, targets = batch_data
                inter_data = torch.from_numpy(inter_data).to(self.device)
                targets = torch.from_numpy(targets).to(self.device)

                outputs = self.tail_model(inter_data)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                await asyncio.sleep(1/1000)

        self.server_logger.info('{} epoch {}, test accuracy {}'.format(
            stage,
            epoch,
            correct/total
        ))

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
                    self.server_logger.info('Received Retraining Config '
                                            'Synchronization request from a '
                                            'client.')
                    if self.config_done:
                        self.server_logger.info('Added Config into the wait'
                                                'list.')
                        self.config_list.append(nxt)
                    else:
                        self.server_logger.info('Proceed to sync Config now.')
                        self.config_sync(nxt)
                elif nxt['msg_type'] == MSG_CODE['inter_data']:
                    self.task_logger.info('Received inter data of split point'
                                          ' {}, batch {}.'.format(
                                              nxt['split_point'],
                                              nxt['batch_num']
                                          ))
                    inter_data = np.reshape(
                        nxt['inter_data'],
                        self.data_shape
                    )
                    classes = nxt['classes']
                    try:
                        self.inter_data_list[nxt['split_point']].append(
                            (inter_data, classes)
                        )
                    except KeyError:
                        print('See data from a new split point')
                        self.inter_data_list[nxt['split_point']] = []
                        self.inter_data_list[nxt['split_point']].append(
                            (inter_data, classes)
                        )
                elif nxt['msg_type'] == MSG_CODE['profile_ready']:
                    self.client_model_load_done = True
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
