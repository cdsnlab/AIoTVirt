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
import torchvision.transforms as transforms
import torchvision
import os
import argparse
from utils import *
from easydict import EasyDict as edict
from dataset.dataloader import PretrainDataset, RetrainDataset
import numpy as np
import random

import matplotlib.pyplot as plt
import pdb
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
    parser.add_argument('-n', '--name', type = str, default = 'resnet18', help = 'name of network')
    parser.add_argument('-a', '--alpha', type = float, default = 0.1, help = 'forgetting hyperparameter. (bigger then faster) (0.1-0.3)')
    parser.add_argument('-t', '--temp', type = int, default = 2, help = 'distillation temperature')
    parser.add_argument('-d', '--dataset', type = str, default = 'cifar10', help = 'name of dataset')
    parser.add_argument('-c', '--new_class', type = str, default = '2,2,2', help = 'number of new classes for each task')
    parser.add_argument('-f', '--finetune_epoch', type = int, default = 70, help = 'the number of finetuning before IL')
    parser.add_argument('-b', '--profile_budget', type = int, default = 3, help = 'time budget for profiling (minute)')
    parser.add_argument('-r', '--retrain_budget', type = int, default = 10, help = 'time budget for retraining (minute)')
    args, _ = parser.parse_known_args()
    
    config = edict()
    config.network = args.name
    config.alpha = args.alpha
    config.temperature = args.temp
    config.dataset = args.dataset
    config.new_class = list(map(int, args.new_class.split(',')))
    config.finetune_epoch = args.finetune_epoch
    config.profile_budget = args.profile_budget*60
    config.retrain_budget = args.retrain_budget*60
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(toRed('\tConfig: {}'.format(config)))
    print(toRed('\tNetwork: {}'.format(config.network)))
    print(toRed('\tDataset : {}'.format(config.dataset)))

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
    
    train_transforms = transforms.Compose([
            # transforms.ToCVImage(),
            # transforms.Resize((64,64)),
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(5),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.Normalize(
                [0.48560741861744905, 0.49941626449353244, 0.43237713785804116],
                [0.2321024260764962, 0.22770540015765814, 0.2665100547329813])
        ])

    test_transforms = transforms.Compose([
            # transforms.ToCVImage(),
            transforms.ToTensor(),
            # transforms.CenterCrop(5),
            transforms.Normalize(
                [0.4862169586881995, 0.4998156522834164, 0.4311430419332438],
                [0.23264268069040475, 0.22781080253662814, 0.26667253517177186])
        ])

    train_dataloaders = []
    test_dataloaders = []
    if config.dataset == 'cifar10':
        task_num = 3
        classes = 4
        for i in range(task_num):
            # cifar10_train_dataset = RetrainDataset(
            #     task_num=i,
            #     dataset_name = config.dataset,
            #     data_dir_path='/data/{}/train'.format(config.dataset),
            #     num_total_classes=10,
            #     num_pretrain_classes=10,
            #     num_test_images_each_class=50,
            #     num_total_images_each_task=16000,
            #     task_specifications=[
            #         (0, 10, 320),
            #         (0, 10, 605),
            #         (0, 10, 570),
            #         # (0, 10, 576),
            #         # (0, 10, 504),
            #         # (0, 10, 604),
            #         # (0, 10, 304)
            #     ],
            #     transforms = train_transforms
            # )
            cifar10_train_dataset = RetrainDataset(
                task_num=i,
                dataset_name = config.dataset,
                data_dir_path='/data/{}/train'.format(config.dataset),
                num_total_classes=10,
                num_pretrain_classes=4,
                num_test_images_each_class=50,
                num_total_images_each_task=10000,
                task_specifications=[
                    (2, 0, 605),
                    (2, 0, 570),
                    (2, 0, 576),
                    # (0, 8, 504),
                    # (2, 0, 604),
                    # (0, 10, 304)
                ],
                transforms = train_transforms
            )
            cifar10_train_dataloader = torch.utils.data.DataLoader(
                cifar10_train_dataset,
                64,
                num_workers = 4,
                shuffle=True
            )
            train_dataloaders.append(cifar10_train_dataloader)
            
            classes += config.new_class[i]
            cifar10_test_dataset = PretrainDataset(
                    dataset_name='cifar10',
                    data_dir_path='/data/cifar10',
                    num_classes_for_pretrain=classes,
                    num_imgs_from_chosen_classes=[
                        (50, classes)
                    ],
                    train=False,
                    choosing_class_seed=2022,
                    train_data_shuffle_seed=223,
                    test_data_shuffle_seed=222,
                    transform = test_transforms
            )
            cifar10_test_dataloader = torch.utils.data.DataLoader(
                cifar10_test_dataset,
                64,
                num_workers = 4,
                shuffle=True
            )
            test_dataloaders.append(cifar10_test_dataloader)
            
    elif config.dataset == 'cifar100':
        task_num = 6
        classes = 100
        for i in range(task_num):
            cifar10_train_dataset = RetrainDataset(
                task_num=i,
                dataset_name = config.dataset,
                data_dir_path='/data/{}/test'.format(config.dataset),
                num_total_classes=100,
                num_pretrain_classes=40,
                num_test_images_each_class=50,
                num_total_images_each_task=1000,
                task_specifications=[
                    (0, 100, 320),
                    (2, 0, 605),
                    (0, 6, 570),
                    (2, 0, 576),
                    (0, 8, 504),
                    (2, 0, 604),
                    (0, 10, 304)
                ],
                transforms = train_transforms
            )
            cifar10_train_dataloader = torch.utils.data.DataLoader(
                cifar10_train_dataset,
                64,
                num_workers = 4,
                shuffle=True
            )
            train_dataloaders.append(cifar10_train_dataloader)
            
            classes += config.new_class[i]
            cifar10_test_dataset = PretrainDataset(
                    dataset_name='cifar100',
                    data_dir_path='/data/cifar100',
                    num_classes_for_pretrain=classes,
                    num_imgs_from_chosen_classes=[
                        (50, classes)
                    ],
                    train=False,
                    choosing_class_seed=2022,
                    train_data_shuffle_seed=223,
                    test_data_shuffle_seed=222,
                    transform = test_transforms
            )
            cifar10_test_dataloader = torch.utils.data.DataLoader(
                cifar10_test_dataset,
                64,
                num_workers = 4,
                shuffle=True
            )
            test_dataloaders.append(cifar10_test_dataloader)

    elif config.dataset == 'imagenet100':
        task_num = 6
        classes = 100
        for i in range(task_num):
            cifar10_train_dataset = RetrainDataset(
                task_num=i,
                dataset_name = config.dataset,
                data_dir_path='/data/{}/test'.format(config.dataset),
                num_total_classes=100,
                num_pretrain_classes=40,
                num_test_images_each_class=50,
                num_total_images_each_task=1000,
                task_specifications=[
                    (0, 100, 320),
                    (2, 0, 605),
                    (0, 6, 570),
                    (2, 0, 576),
                    (0, 8, 504),
                    (2, 0, 604),
                    (0, 10, 304)
                ],
                transforms = train_transforms
            )
            cifar10_train_dataloader = torch.utils.data.DataLoader(
                cifar10_train_dataset,
                64,
                num_workers = 4,
                shuffle=True
            )
            train_dataloaders.append(cifar10_train_dataloader)
            
            classes += config.new_class[i]
            cifar10_test_dataset = PretrainDataset(
                    dataset_name='imagenet100',
                    data_dir_path='/data/imagenet10',
                    num_classes_for_pretrain=classes,
                    num_imgs_from_chosen_classes=[
                        (50, classes)
                    ],
                    train=False,
                    choosing_class_seed=2022,
                    train_data_shuffle_seed=223,
                    test_data_shuffle_seed=222,
                    transform = test_transforms
            )
            cifar10_test_dataloader = torch.utils.data.DataLoader(
                cifar10_test_dataset,
                64,
                num_workers = 4,
                shuffle=True
            )
            test_dataloaders.append(cifar10_test_dataloader)

    '''
    Calculate the retrain times in every split points.
    The batch size of each dataloaders has to be same.
    The computation time and network time can be differenct if batch size is changed.
    To insult equal number of lables, use the pretrain dataset.
    '''

    # cifar10_train_dataset = PretrainDataset(
    #     dataset_name='cifar10',
    #     data_dir_path='/data/cifar10',
    #     num_classes_for_pretrain=1,
    #     num_imgs_from_chosen_classes=[
    #         (128, 1)
    #     ],
    #     train=True,
    #     choosing_class_seed=2022,
    #     train_data_shuffle_seed=223,
    #     test_data_shuffle_seed=222,
    #     transform = train_transforms
    # )

    # cifar10_train_dataloader = torch.utils.data.DataLoader(
    #     cifar10_train_dataset,
    #     64,
    #     num_workers = 4,
    #     shuffle=True
    # )

    # trainer.measure_latency(dataloader=train_dataloaders[0])
    ######################################## dataset load ########################################
    
    
    ######################################## golden model ########################################
    golden_model = None
    ######################################## golden model ########################################
    
    
    trainer.set_network(split_point=0)
    

    # cifar10_test_dataset = PretrainDataset(
    #         dataset_name='cifar10',
    #         data_dir_path='/data/cifar10',
    #         num_classes_for_pretrain=4,
    #         num_imgs_from_chosen_classes=[
    #             (50, 4)
    #         ],
    #         train=False,
    #         choosing_class_seed=2022,
    #         train_data_shuffle_seed=223,
    #         test_data_shuffle_seed=222,
    #         transform = test_transforms
    #     )
    # cifar10_test_dataloader = torch.utils.data.DataLoader(
    #     cifar10_test_dataset,
    #     64,
    #     num_workers = 4,
    #     shuffle=True
    # )
    # trainer.test(dataloader = cifar10_test_dataloader, num_task = 0, epoch = 0)
    # trainer.test(dataloader = cifar10_test_dataloader, num_task = 0, epoch = 1)


    # print(trainer.tail_model.model)


    for dataloader_idx in range(len(train_dataloaders)):
        dataloader = train_dataloaders[dataloader_idx]
        test_dataloader = test_dataloaders[dataloader_idx]
        num_new_class = config.new_class[dataloader_idx]
        print(toYellow('######### Retrain Start Task {} #########'.format(dataloader_idx)))
        trainer.incremental_learning(dataloader=dataloader, test_dataloader=test_dataloader, epoch=70,
                                    num_new_class=num_new_class, num_task=dataloader_idx)
        trainer.save_network(dataset = config.dataset, num_task = dataloader_idx)











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
    
    allocated_profile_time = config.profile_budget / totallayer[config.network]
    for dataloader_idx in range(len(train_dataloaders)):
        dataloader = train_dataloaders[dataloader_idx]
        test_dataloader = test_dataloaders[dataloader_idx]
        num_new_class = config.new_class[dataloader_idx]
        retrain_results = dict()
        print(toYellow('######### Profile Start Task {} #########'.format(dataloader_idx)))
        # for split_point in range(totallayer[config.network]):
        #     '''
        #     Profiling phase start.
        #     The number of epochs to be used in profiling phase at the split point is calculated.
        #     And measure the accuracy based on split point and calculated epochs.
        #     '''
        #     time_train_one_step = trainer.get_time_train_one_step(split_point=split_point)
        #     # time_train_one_epoch = math.ceil(len(dataloader.dataset)/64)*time_train_one_step
        #     time_train_one_epoch = time_train_one_step
        #     number_of_profile_epoch = int(allocated_profile_time / time_train_one_epoch)
        #     print(toBlue('task num : {},\t'.format(dataloader_idx)) + toCyan('split point : {},\t'.format(split_point))
        #      + toGreen('train one epoch time : {},\t'.format(time_train_one_epoch))
        #      + toMagenta('allocated time : {},\t'.format(allocated_profile_time))
        #      + toRed('allocated epoch : {},\t'.format(number_of_profile_epoch)))
        #     trainer.set_network(split_point=split_point)
        #     retrain_results[split_point] = trainer.incremental_learning(dataloader=dataloader, test_dataloader=test_dataloader,
        #                                     epoch=number_of_profile_epoch, num_new_class=num_new_class if split_point == 0 else 0,
        #                                     num_task=dataloader_idx, is_profile=True)

        # '''
        # After profiling phase finish, then split the model and execute IL until accuracy converges.
        # And collect the datas for diagram.
        # '''
        # print(retrain_results)
        # # sorted_retrain_results = sorted(retrain_results.items(), key=lambda x: x[0], reverse=True)
        # best_idx, best_val = 0, 0.
        # for i in range(totallayer[config.network]):
        #     if best_val < retrain_results[i]:
        #         best_val = retrain_results[i]
        #         best_idx = i
        # best_split_train_acc = best_val
        # best_split_point = best_idx
        # trainer.set_network(split_point=best_split_point)
        trainer.set_network(split_point=0)

        '''
        Similarly with profiling phase, the number of retraining epoch is decided 
        based on retrain budget which is defined by user.
        '''
        # IL the model
        # time_train_one_step = trainer.get_time_train_one_step(split_point=best_split_point)
        time_train_one_step = trainer.get_time_train_one_step(split_point=0)
        # time_train_one_epoch = math.ceil(len(dataloader.dataset)/dataloader.batchsize)*time_train_one_step
        time_train_one_epoch = time_train_one_step
        number_of_retrain_epoch = int(config.retrain_budget / time_train_one_epoch)
        print(toYellow('######### Retrain Start Task {} #########'.format(dataloader_idx)))
        # print(toBlue('task num : {},\t'.format(dataloader_idx)) + toCyan('split point : {},\t'.format(best_split_point))
        print(toBlue('task num : {},\t'.format(dataloader_idx)) + toCyan('split point : {},\t'.format(0))
            + toGreen('train one epoch time : {},\t'.format(time_train_one_epoch))
            + toMagenta('allocated time : {},\t'.format(config.retrain_budget))
            + toRed('allocated epoch : {},\t'.format(number_of_retrain_epoch)))
        trainer.incremental_learning(dataloader=dataloader, test_dataloader = test_dataloader, 
                                    epoch=number_of_retrain_epoch, num_new_class=num_new_class, num_task=dataloader_idx, is_profile=False)
        # test the model
        # best_split_test_acc = trainer.test()

        # total_history.append([best_split_point, best_split_train_acc, best_split_test_acc])

    