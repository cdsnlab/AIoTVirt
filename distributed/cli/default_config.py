import os
import sys
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
    from transforms import RandomHorizontalFlip, ColorJitter, NormalizeNumpy,\
        Compose
except ModuleNotFoundError as e:
    raise e


class config:

    server_ip = '143.248.55.76'
    server_port = '5001'

    model_name = 'resnet18'
    dev = True

    split_point_list = np.arange(10)

    dataset_name = 'cifar10'
    data_dir_path = '/data/cifar10'
    interdata_shape_dict_path = \
        '/home/ligeti/models/source/interdata_shape_dict.json'
    log_config_path = 'logs/log_config.json'
    
    learning_rate = 1e-6
    num_classes_for_pretrain = 10
    num_imgs_from_chosen_pretrain_classes = [
        (500, 3), (1000, 3), (1500, 2), (2000, 2)
    ]
    num_imgs_from_chosen_test_classes = [
        (50, 10)
    ]
    choosing_class_seed = 2022
    pretrain_train_data_shuffle_seed = 223
    pretrain_test_data_shuffle_seed = 222
    task_specifications = [
        (10, 10000, 810)
    ],
    retrain_data_shuffle_seed = 2
    batch_size = 64
    img_height = 32
    img_width = 32
    fp = 16

    transforms = Compose([
        RandomHorizontalFlip(),
        # ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
        NormalizeNumpy(
            [0.48560741861744905, 0.49941626449353244, 0.43237713785804116],
            [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
        )
    ])
    test_transform = Compose([
        NormalizeNumpy(
            [0.48560741861744905, 0.49941626449353244, 0.43237713785804116],
            [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
        )
    ])
