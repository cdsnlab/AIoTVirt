import torch
from torchvision import datasets
import torchvision
import copy
import torch.nn as nn
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

'''
For initializing EfficientNet.
'''
def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, nonlinearity='sigmoid')

fclayer = {'resnet18': 1, 'resnet34': 1, 'resnet50': 1, 'resnet101': 1,
            'resnext50': 1, 'resnext101': 1,
            'vgg11': 7, 'vgg13': 7, 'vgg16': 7, 'vgg19': 7,
            'mobilenetv2': 2, 'shufflenetv2': 1, 'alexnet': 7, 'googlenet': 1, 'efficientnet_b0': 1}
totallayer = {'resnet18': 14, 'resnet34': 22, 'resnet50': 22, 'resnet101': 39,
                'resnext50': 22, 'resnext101': 39,
                'vgg11': 29, 'vgg13': 33, 'vgg16': 39, 'vgg19': 45,
                'mobilenetv2': 21, 'shufflenetv2': 24, 'alexnet': 21, 'googlenet': 18, 'efficientnet_b0': 8}
num_lastlayer = {'cifar10': 4, 'cifar100': 40, 'imagenet100': 40}
# num_lastlayer = {'cifar10': 10, 'cifar100': 100, 'imagenet100': 100}

def model_spec(model_name, dataset_name):
    input_transform = None
    # directory = '/app/videosample/CRED/'

    if model_name=='resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # if not alternative:
        #     model.load_state_dict(torch.load(directory+'resnet18_trained_57_2.pt'), strict=False)
        # else:
        #     model.load_state_dict(torch.load(directory + 'alternative_resnet18.pt'), strict=False)
        fc = model.fc
    if model_name=='resnet34':
        model = torchvision.models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'resnet34_trained_58.pt'), strict=False)
        fc = model.fc
    if model_name=='resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'resnet50_trained_62.pt'), strict=False)
        fc = model.fc
    if model_name=='resnet101':
        model = torchvision.models.resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'resnet101_trained_60.pt'), strict=False)
        fc = model.fc
    if model_name=='resnext50':
        model = torchvision.models.resnext50_32x4d(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'resnext50_trained_61.pt'), strict=False)
        fc = model.fc
    if model_name=='resnext101':
        model = torchvision.models.resnext101_32x8d(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'resnext101_trained_63.pt'), strict=False)
        fc = model.fc
    if model_name=='vgg11':
        model = torchvision.models.vgg11(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # if not alternative:
        #     model.load_state_dict(torch.load(directory+'vgg11_trained_60.pt'))
        # else:
        #     model.load_state_dict(torch.load(directory + 'alternative_vgg11.pt'), strict=False)

        fc = model.classifier
    if model_name=='vgg13':
        model = torchvision.models.vgg13(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'vgg13_trained_67.pt'))
        fc = model.classifier
    if model_name=='vgg16':
        model = torchvision.models.vgg16(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'vgg16_trained_67.pt'))
        fc = model.classifier
    if model_name=='vgg19':
        model = torchvision.models.vgg19(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'vgg19_trained_67.pt'))
        fc = model.classifier
    if model_name=='mobilenetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_lastlayer[dataset_name])
        # if not alternative:
        #     model.load_state_dict(torch.load(directory+'mobilenetv2_trained_57.pt'), strict=False)
        # else:
        #     model.load_state_dict(torch.load(directory + 'alternative_mobilenetv2.pt'), strict=False)
        fc = model.classifier
    if model_name=='shufflenetv2':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'shufflenetv2_trained_54_3.pt'), strict=False)
        fc = model.fc
    if model_name=='alexnet':
        model = torchvision.models.alexnet()
        # model.load_state_dict(torch.load(directory+'alexnet_trained_54_3.pt'), strict=False)
        fc = model.classifier
    if model_name=='googlenet':
        model = torchvision.models.googlenet(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_lastlayer[dataset_name])
        # model.load_state_dict(torch.load(directory+'googlenet_trained_72.pt'), strict=True)
        fc = model.fc
        input_transform = model._transform_input
    if model_name=='efficientnet_b0':
        # model = torchvision.models.efficientnet_b0(pretrained=False)
        model = EfficientNet.from_pretrained('efficientnet-b0')
        xavier_normal_init(model)
        model._fc = torch.nn.Linear(model._fc.in_features, num_lastlayer[dataset_name])
        fc = model._fc

    return model, fc, fclayer[model_name], totallayer[model_name], input_transform
