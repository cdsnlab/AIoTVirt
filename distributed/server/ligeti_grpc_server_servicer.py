import os
import sys
from collections import deque
from pickle import loads
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
sys.path.append(BASE_DIR)

try:
    import ligeti_inter_data_pb2 as ligeti_grpc_msg
    import ligeti_inter_data_pb2_grpc as ligeti_grpc_server
    from google.protobuf.timestamp_pb2 import Timestamp
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
        classes = loads(request.classes).astype('int64')
        is_recv_successful = 0
        if isinstance(inter_data, np.ndarray):
            is_recv_successful = 1

        msg_time = request.timestamp
        msg_time = msg_time.seconds + msg_time.nanos / 1000000.

        self.inbound_queue.append({
            'msg_type': MSG_CODE['inter_data'],
            'split_point': request.split_point,
            'batch_num': request.batch_num,
            'inter_data': inter_data,
            'classes': classes,
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
        data_shape = (
            -1,
            request.data_shape.num_channels,
            request.data_shape.height,
            request.data_shape.width
        )
        config = {
            'msg_type': MSG_CODE['config_sync'],
            'model_name': request.model_name,
            'dataset_name': request.dataset_name,
            'num_classes': request.num_classes,
            'num_batches': request.num_batches,
            'batch_size': request.batch_size,
            'split_point': request.split_point,
            'lr': request.lr,
            'data_shape': data_shape,
            'temperature': request.temperature,
            'alpha': request.alpha,
            'num_profile_epochs': request.num_profile_epochs,
            'num_retrain_epochs': request.num_retrain_epochs,
            'retrain_time_budget': request.retrain_time_budget,
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