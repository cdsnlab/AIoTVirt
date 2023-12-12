import argparse
import yaml
from easydict import EasyDict
from typing import Dict, Callable, Tuple, Union, List


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, 
                        help= "path for learnt parameters saving")
    
    # Default is in configs/config.yaml

    parser.add_argument("--model", type=str, default=None, help='model name')
    # random seed
    parser.add_argument("--random_seed", type=int, default=None,
                        help="random seed (default: 1)")
                        
    parser.add_argument("--exp_name", type=str, help = "experiment name")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_order", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--adapt_bn", action='store_true', default=False)
    parser.add_argument("--no_adapt", action='store_true', default=False)
    parser.add_argument("--backward_free", action='store_true', default=False)
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--infer_test", action='store_true', default=False)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--short_test", type=int, default=-1)
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--input_size", type=str, default=None)
    parser.add_argument("--eval_clean", action='store_true', default=False)

    return parser


def get_config(args: argparse.Namespace, config_path='configs/config.yaml') -> argparse.Namespace:
    if config_path.split('.')[-1] != 'yaml':
        config_path = f'configs/{config_path}.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)
        config = EasyDict(config)

    # copy parsed arguments
    for key in args.__dir__():
        if key[:2] != '__':
            if getattr(args, key) is not None or not hasattr(config, key):
                setattr(config, key, getattr(args, key))

    with open('configs/data.yaml') as f:
        data_conf = yaml.safe_load(f)
        config.datasets = data_conf['datasets']

    print(f"Config loaded from {config_path}")
    return config


def get_opts(config_path='configs/config.yaml') -> EasyDict:
    opts = get_argparser().parse_args()
    return merge_opts(opts, config_path)

# opts.config is preferred to config_path
def merge_opts(opts: Dict, config_path='configs/config.yaml') -> argparse.Namespace:
    return get_config(opts, config_path=opts.config if opts.config else config_path)


def execute_by(branches: Dict[str, Tuple[Callable[[EasyDict], any]]], by='task'):
    opts = get_argparser().parse_args()
    if hasattr(opts, by):
        task = getattr(opts, by)
        if task in branches:
            fn = branches[task]
            if type(fn) == tuple:
                fn, config = branches[task]
            opts = merge_opts(opts, config_path=config)
            return fn(opts)
        
        else:
            raise NotImplementedError(f"Task {by}={task} is not implemented")
    
    raise NotImplementedError(f"{by} is not in the opts")

