import argparse
import yaml
from easydict import EasyDict


def get_config(args: argparse.Namespace, meta=True) -> EasyDict:
    if args.stage == 0:
        if meta:
            config_path = 'configs/train-meta.yaml'
        else:
            config_path = 'configs/train-nonmeta.yaml'
    elif args.stage == 1:
        if meta:
            config_path = 'configs/finetune-meta.yaml'
        else:
            config_path = 'configs/finetune-nonmeta.yaml'
    elif args.stage == 2:
        config_path = 'configs/test.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)

    # copy parsed arguments
    for key in args.__dir__():
        if key[:2] != '__' and getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))

    with open('configs/data.yaml') as f:
        data_conf = yaml.safe_load(f)
        config.data_root = data_conf['data_root']
        config.datasets = data_conf['datasets']

    if 'task' not in config:
        from dataset.utils import TASK_DATASETS_TEST, TASK_DATASETS_TRAIN
        config.task = TASK_DATASETS_TEST[config.case] if config.stage > 1 else TASK_DATASETS_TRAIN

    if 'checkpoint' not in config:
        config.checkpoint = None

    return config
