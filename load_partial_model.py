import torch
from torchvision import datasets
import torchvision
import copy
import torch.nn as nn
# from models.archs.
from efficientnet_pytorch import EfficientNet


# model = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101,
#          'resnext50': resnext50, 'resnext101': resnext101,
#          'vgg11': vgg11, 'vgg13': vgg13, 'vgg16': vgg16, 'vgg19': vgg19,
#          'alexnet': alexnet, 'googlenet': googlenet,
#          'mobilenetv2': mobilenetv2,
#          'shufflenetv2': shufflenetv2}
# fc = {'resnet18': resnet18.fc, 'resnet34': resnet34.fc, 'resnet50': resnet50.fc, 'resnet101': resnet101.fc,
#       'resnext50': resnext50.fc, 'resnext101': resnext101.fc,
#       'vgg11': vgg11.classifier, 'vgg13': vgg13.classifier, 'vgg16': vgg16.classifier, 'vgg19': vgg19.classifier,
#       'alexnet': alexnet.classifier, 'googlenet': googlenet.fc,
#       'mobilenetv2': mobilenetv2.classifier, 'shufflenetv2': shufflenetv2.fc}


fclayer = {'resnet18': 1, 'mobilenetv2': 1, 'googlenet': 1, 'efficientnet_b0': 1}
totallayer = {'resnet18': 14, 'mobilenetv2': 22, 'googlenet': 21, 'efficientnet_b0': 9}
# num_lastlayer = {'cifar10': 4, 'cifar100': 40, 'imagenet100': 40}
num_lastlayer = {'cifar10': 10, 'cifar100': 100, 'imagenet100': 100}

def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, nonlinearity='sigmoid')

def model_spec(model_name, dataset_name):
    input_transform = None
    # directory = '/app/videosample/CRED/'

    if model_name=='resnet18':
        # model = ResNet18(pretrained=False)
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        fc = model.fc
    
    if model_name=='mobilenetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_lastlayer[dataset_name])
        fc = model.classifier
    if model_name=='googlenet':
        model = torchvision.models.googlenet(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        fc = model.fc
        input_transform = model._transform_input
    if model_name=='efficientnet_b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        xavier_normal_init(model)
        model._fc = torch.nn.Linear(model._fc.in_features, num_lastlayer[dataset_name])
        fc = model._fc

    return model, fc, fclayer[model_name], totallayer[model_name], input_transform
