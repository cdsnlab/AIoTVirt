from dataset import dataloader
from load_partial_model import model_spec
from dataloader import PretrainDataset
from torch.autograd import Variable
from utils import toGreen, toRed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

'''
Pretrain below 4 models for 100 epochs at each model.
And Each model is trained with below 3 datasets.
Several augmentation methods are used in training phase.
'''
directory = './ckpt/pretrain/'
models = ['resnet18', 'googlenet', 'mobilenetv2', 'efficientnet_b0']

datasets = ['cifar10', 'cifar100', 'imagenet100']
train_dataloaders = []
test_dataloaders = []

train_transforms = transforms.Compose([
        # transforms.ToCVImage(),
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

'''
Load the datasets and dataloaders.
And store them in list.
'''
cifar10_train_dataset = PretrainDataset(
        dataset_name='cifar10',
        data_dir_path='/data/cifar10',
        num_classes_for_pretrain=4,
        num_imgs_from_chosen_classes=[
            (500, 1), (1000, 1), (1500, 1), (2000, 1)
        ],
        train=True,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222,
        transform = train_transforms
    )
cifar10_test_dataset = PretrainDataset(
        dataset_name='cifar10',
        data_dir_path='/data/cifar10',
        num_classes_for_pretrain=4,
        num_imgs_from_chosen_classes=[
            (50, 4)
        ],
        train=False,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222,
        transform = test_transforms
    )
cifar10_train_dataloader = torch.utils.data.DataLoader(
    cifar10_train_dataset,
    64,
    num_workers = 4,
    shuffle=True
)
cifar10_test_dataloader = torch.utils.data.DataLoader(
    cifar10_test_dataset,
    64,
    num_workers = 4,
    shuffle=True
)
train_dataloaders.append(cifar10_train_dataloader)
test_dataloaders.append(cifar10_test_dataloader)

cifar100_train_dataset = PretrainDataset(
        dataset_name='cifar100',
        data_dir_path='/data/cifar100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (50, 10), (150, 20), (200, 10)
        ],
        train=True,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222,
        transform = train_transforms
    )
cifar100_test_dataset = PretrainDataset(
        dataset_name='cifar100',
        data_dir_path='/data/cifar100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (50, 40)
        ],
        train=False,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222,
        transform = test_transforms
    )
cifar100_train_dataloader = torch.utils.data.DataLoader(
    cifar100_train_dataset,
    64,
    num_workers = 4,
    shuffle=True
)
cifar100_test_dataloader = torch.utils.data.DataLoader(
    cifar100_test_dataset,
    64,
    num_workers = 4,
    shuffle=True
)
train_dataloaders.append(cifar100_train_dataloader)
test_dataloaders.append(cifar100_test_dataloader)

imagenet100_train_dataset = PretrainDataset(
        dataset_name='imagenet100',
        data_dir_path='/data/imagenet100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (100, 10), (250, 20), (500, 10)
        ],
        train=True,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222,
        transform = train_transforms
    )
imagenet100_test_dataset = PretrainDataset(
        dataset_name='imagenet100',
        data_dir_path='/data/imagenet100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (50, 40)
        ],
        train=False,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222,
        transform = test_transforms
    )
imagenet100_train_dataloader = torch.utils.data.DataLoader(
    imagenet100_train_dataset,
    64,
    num_workers = 4,
    shuffle=True
)
imagenet100_test_dataloader = torch.utils.data.DataLoader(
    imagenet100_test_dataset,
    64,
    num_workers = 4,
    shuffle=True
)
train_dataloaders.append(imagenet100_train_dataloader)
test_dataloaders.append(imagenet100_test_dataloader)


'''
Pretraining start.
In each dataset and model setting, the models are trained for 100 epochs.
And save them in './ckpt/pretrain' directory.
'''
for i in range(len(train_dataloaders)):
    print(toRed(datasets[i]))
    train_dataloader = train_dataloaders[i]
    test_dataloader = test_dataloaders[i]
    
    for name in models:
        print(toGreen(name))
        model, _, _, _, input_transform = model_spec(name, datasets[i])
        model.cuda(0)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=0.001)

        epoch = 0
        before_eval_acc = 0.

        # epoch
        while epoch < 100:
            epoch += 1
            total_right = 0
            total = 0
            
            model.train()
            for inputs, labels, _ in train_dataloader:
                inputs, labels = Variable(inputs.float()).cuda(0), Variable(labels).cuda(0)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                if name == 'googlenet':
                    outputs = outputs[0]

                loss = loss_fn(outputs,labels)
                loss.backward()
                optimizer.step()
                
                predicted = outputs.data.max(1)[1]
                # print(predicted)
                total += labels.size(0)
                total_right += (predicted == labels.data).float().sum()
            
            train_acc = 100 * total_right / total
            if epoch % 10 == 0:
                print("Model: {}, Training accuracy for epoch {} : {}".format(name, str(epoch), train_acc))
            
            total_right = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for data in test_dataloader:
                    images, labels, _ = data
                    images, labels = Variable(images.float()).cuda(0),Variable(labels).cuda(0)
                    
                    images = F.interpolate(images,[224,224])
                    
                    outputs = model(images)
                
                    _, predicted = torch.max(outputs.data,1)
                    total += labels.size(0)
                    total_right += (predicted == labels.data).float().sum()
            
            eval_acc = 100 * total_right / total
            if epoch % 10 == 0:
                print("Test accuracy after retrain: {}".format(eval_acc))
            
            # if before_eval_acc > eval_acc and train_acc - 10 < eval_acc:
            #     torch.save(model.state_dict(), directory + name + '.pt')
            #     break
            # before_eval_acc = total_right/total

        torch.save(model.state_dict(), directory + name + '_' + datasets[i] + '.pt')