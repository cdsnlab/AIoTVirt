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

