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