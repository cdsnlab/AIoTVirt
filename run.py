'''
The split point profiler and IL(Incremental Learning) traniners are run in this code.
You can set the configurations using the command line arguments.
First, load the dataloaders that consist of serveral tasks.
Second, profile the split point.
Last, split the model based on profiled split point, and makes IL in the splitted model.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.io as scio
from scipy.io import loadmat
import torchvision
import os
import argparse
from utils import *
from easydict import EasyDict as edict
import numpy as np
import random

import matplotlib.pyplot as plt
import pdb
import transforms
from dataset import CUB_200_2011_Train, CUB_200_2011_Test
import torchvision.transforms as tfs
import torchvision.datasets as datasets
from LwF_trainer import Trainer

'''
The information of layer in each model is listed in dictionary.
These are used when splitting the model.
'''
fclayer = {'resnet18': 1, 'resnet34': 1, 'resnet50': 1, 'resnet101': 1,
            'resnext50': 1, 'resnext101': 1,
            'vgg11': 7, 'vgg13': 7, 'vgg16': 7, 'vgg19': 7,
            'mobilenetv2': 2, 'shufflenetv2': 1, 'alexnet': 7, 'googlenet': 1}
totallayer = {'resnet18': 14, 'resnet34': 22, 'resnet50': 22, 'resnet101': 39,
            'resnext50': 22, 'resnext101': 39,
            'vgg11': 29, 'vgg13': 33, 'vgg16': 39, 'vgg19': 45,
            'mobilenetv2': 21, 'shufflenetv2': 24, 'alexnet': 21, 'googlenet': 18}

if __name__ == '__main__':

    '''
    You can reproduce the results, when the random seed should be same.
    '''
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    Configurations can be set simply by using only command line argument.
    Also, the use of shell script is recommended.
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', action = str, default = 'resnet18', help = 'name of network')
    parser.add_argument('-a', '--alpha', action = int, default = 0.1, help = 'forgetting hyperparameter. (bigger then faster) (0.1-0.3)')
    parser.add_argument('-t', '--temp', action = int, default = 2, help = 'distillation temperature')
    parser.add_argument('-d', '--dataset', action = int, default = 'cifar10', help = 'name of dataset')
    parser.add_argument('-b', '--profile_budget', action = int, default = 20, help = 'time budget for profiling (minute)')
    parser.add_argument('-r', '--retrain_budget', action = int, default = 20, help = 'time budget for retraining (minute)')
    args, _ = parser.parse_known_args()
    
    config = edict()
    config.network = args.name
    config.alpha = args.alpha
    config.temperature = args.temp
    config.dataset = args.dataset
    config.profile_budget = args.profile_budget*60
    config.retrain_budget = args.retrain_budget*60
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(toRed('\tConfig: {}'.format(config.config)))
    print(toRed('\tNetwork: {}'.format(config.network)))
    print(toRed('\tDataset : {}'.format(config.data)))

    '''
    Define the Trainer that consists of head model and tail model.
    This can split the model based on split_point parameter, and make the model learn incrementally.
    '''
    trainer = Trainer(config)


    '''
    Load the dataloaders.
    Each dataloader has different task.
    i.e., they go into the model sequentially to make data drift.
    And calculate the retrain times in every split points.
    '''
    ######################################## dataset load ########################################
    dataloaders = []
    for i in range(4):
        dataloader = None
        dataloaders.append(dataloader)
    
    '''
    Calculate the retrain times in every split points.
    The batch size of each dataloaders has to be same.
    The computation time and network time can be differenct if batch size is changed.
    '''
    trainer.measure_latency(dataloader=dataloaders[0])
    ######################################## dataset load ########################################
    
    
    ######################################## golden model ########################################
    golden_model = None
    ######################################## golden model ########################################
    
    
    '''
    The dataloaders are used seuentially.
    In each dataloader, the optimal split point is profiled.
    To fairly profile, the same time should be allocated to each split point in profile phase.
    Time budget that was set via argument divided by the number of layer in network is allocated_time in each split point.
    The allowed epochs in each split points are calculated by dividing the allocated_time by time_train_onestep 
    that consists of the sum of forward propagation in head model and tail model and back propagation in tail model.
    Decide the optimal split point based on profiled information.
    Finally, split the model and make IL in splitted model until accuracy converges.
    '''
    # for diagram
    total_history = []
    
    allocated_time = config.profile_budget / totallayer[config.network]
    for dataloader in dataloaders:
        num_new_class = 1
        retrain_results = dict()
        for split_point in range(totallayer[config.network]):
            '''
            Profiling phase start.
            The number of epochs to be used in profiling phase at the split point is calculated.
            And measure the accuracy based on split point and calculated epochs.
            '''
            time_train_one_step = trainer.get_time_train_one_step(split_point=split_point)
            time_train_one_epoch = math.ceil(len(dataloader.dataset)/dataloader.batchsize)*time_train_one_step
            number_of_profile_epoch = int(allocated_time / time_train_one_epoch)
            trainer.set_network(split_point=split_point)
            retrain_results[split_point], _ = trainer.incremental_learning(dataloader=dataloader, 
                                                                        epoch=number_of_profile_epoch, num_new_class=num_new_class, is_profile=True)
        
        '''
        After profiling phase finish, then split the model and execute IL until accuracy converges.
        And collect the datas for diagram.
        '''
        sorted_retrain_results = sorted(retrain_results.items(), key=lambda x: x[0], reverse=True)
        best_split_train_acc = sorted_retrain_results.values()[0]
        best_split_point = sorted_retrain_results.keys()[0]
        trainer.set_network(split_point=best_split_point)
        
        '''
        Similarly with profiling phase, the number of retraining epoch is decided 
        based on retrain budget which is defined by user.
        '''
        # IL the model
        time_train_one_step = trainer.get_time_train_one_step(split_point=best_split_point)
        time_train_one_epoch = math.ceil(len(dataloader.dataset)/dataloader.batchsize)*time_train_one_step
        number_of_retrain_epoch = int(config.retrain_budget / time_train_one_epoch)
        trainer.incremental_learning(dataloader=dataloader, epoch=number_of_retrain_epoch, num_new_class=num_new_class)
        # test the model
        best_split_test_acc = trainer.test()

        total_history.append([best_split_point, best_split_train_acc, best_split_test_acc])

    