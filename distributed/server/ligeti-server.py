import os
import sys
import grpc
import asyncio
import numpy as np
import json
from collections import deque
from pickle import dumps, loads
import ligeti_inter_data_pb2 as ligeti_grpc_msg
import ligeti_inter_data_pb2_grpc as ligeti_grpc_server
from google.protobuf.timestamp_pb2 import Timestamp
import time
import torch

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


class LigetiGrpcServicer(ligeti_grpc_server.LigetiProfileServicer):
    def __init__(self):
        self.inbound_queue = deque()

    async def inter_data_stream(self, request, context):
        current_time = Timestamp()
        current_time.GetCurrentTime()
        inter_data = loads(request.inter_data)
        is_recv_successful = 0
        if isinstance(inter_data, np.ndarray):
            is_recv_successful = 1

        msg_time = request.timestamp
        msg_time = msg_time.seconds + msg_time.nanos / 1000000.
        data_shape = (
            -1,
            request.data_shape.num_channels,
            request.data_shape.height,
            request.data_shape.width
        )

        self.inbound_queue.append({
            'msg_type': MSG_CODE['inter_data'],
            'split_point': request.split_point,
            'batch_num': request.batch_num,
            'inter_data': inter_data,
            'data_shape': data_shape,
            'msg_time': msg_time
        })

        resp = ligeti_grpc_msg.InterDataRecvConfirm(
            msg_type=MSG_CODE['inter_data_confirm'],
            split_point=request.split_point,
            batch_num=request.batch_num,
            ack=is_recv_successful,
            timestamp=current_time
        )
        return resp

    async def config_sync(self, request, context):
        config = {
            'msg_type': MSG_CODE['config_sync'],
            'model_name': request.model_name,
            'dataset_name': request.dataset_name,
            'num_classes': request.num_classes,
            'num_batches': request.num_batches,
            'batch_size': request.batch_size,
            'split_point': request.split_point,
            'lr': request.lr
        }
        self.inbound_queue.append(config)
        resp = ligeti_grpc_msg.ConfigConfirm(
            msg_type=MSG_CODE['config_sync_confirm'],
            ack=1
        )
        return resp

    async def profile_ready_signal(self, request, context):
        resp = ligeti_grpc_msg.ProfileReadyConfirm(
            msg_type=MSG_CODE['profile_ready_confirm']
        )
        self.inbound_queue.append(
            {
                'msg_type': MSG_CODE['profile_ready']
            }
        )
        return resp

class LigetiServer():
    def __init__(
        self,
        device='cuda:0',
        port=5001
    ):
        self.port = port
        self.running_mode = 'profile'
        self.start_profiling = True
        self.device = device
        self.inter_data_list = {}
        self.tail_model = None
        self.config_done = False
        self.client_model_convert_done = False
        self.config_list = deque()

    def ready_to_profile(self):
        return self.client_model_convert_done and self.config_done

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
        self.grpc_server.add_insecure_port('[::]:{}'.format(self.port))

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
        print('Set config {}'.format(config))

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
