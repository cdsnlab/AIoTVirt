import torch
from torchvision import datasets
import torchvision
import copy

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
fclayer = {'resnet18': 1, 'resnet34': 1, 'resnet50': 1, 'resnet101': 1,
            'resnext50': 1, 'resnext101': 1,
            'vgg11': 7, 'vgg13': 7, 'vgg16': 7, 'vgg19': 7,
            'mobilenetv2': 2, 'shufflenetv2': 1, 'alexnet': 7, 'googlenet': 1}
totallayer = {'resnet18': 14, 'resnet34': 22, 'resnet50': 22, 'resnet101': 39,
                'resnext50': 22, 'resnext101': 39,
                'vgg11': 29, 'vgg13': 33, 'vgg16': 39, 'vgg19': 45,
                'mobilenetv2': 21, 'shufflenetv2': 24, 'alexnet': 21, 'googlenet': 18}


def model_spec(model_name, alternative):
    input_transform = None
    directory = '/app/videosample/CRED/'

    if model_name=='resnet18':
        model = torchvision.models.resnet18()
        model.fc = torch.nn.Linear(512, 100)
        if not alternative:
            model.load_state_dict(torch.load(directory+'resnet18_trained_57_2.pt'), strict=False)
        else:
            model.load_state_dict(torch.load(directory + 'alternative_resnet18.pt'), strict=False)
        fc = model.fc
    if model_name=='resnet34':
        model = torchvision.models.resnet34()
        model.fc = torch.nn.Linear(512, 100)
        model.load_state_dict(torch.load(directory+'resnet34_trained_58.pt'), strict=False)
        fc = model.fc
    if model_name=='resnet50':
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048, 100)
        model.load_state_dict(torch.load(directory+'resnet50_trained_62.pt'), strict=False)
        fc = model.fc
    if model_name=='resnet101':
        model = torchvision.models.resnet101()
        model.fc = torch.nn.Linear(2048, 100)
        model.load_state_dict(torch.load(directory+'resnet101_trained_60.pt'), strict=False)
        fc = model.fc
    if model_name=='resnext50':
        model = torchvision.models.resnext50_32x4d()
        model.fc = torch.nn.Linear(2048, 100)
        model.load_state_dict(torch.load(directory+'resnext50_trained_61.pt'), strict=False)
        fc = model.fc
    if model_name=='resnext101':
        model = torchvision.models.resnext101_32x8d()
        model.fc = torch.nn.Linear(2048, 100)
        model.load_state_dict(torch.load(directory+'resnext101_trained_63.pt'), strict=False)
        fc = model.fc
    if model_name=='vgg11':
        model = torchvision.models.vgg11()
        model.classifier[-1] = torch.nn.Linear(4096, 100)
        if not alternative:
            model.load_state_dict(torch.load(directory+'vgg11_trained_60.pt'))
        else:
            model.load_state_dict(torch.load(directory + 'alternative_vgg11.pt'), strict=False)

        fc = model.classifier
    if model_name=='vgg13':
        model = torchvision.models.vgg13()
        model.classifier[-1] = torch.nn.Linear(4096, 100)
        model.load_state_dict(torch.load(directory+'vgg13_trained_67.pt'))
        fc = model.classifier
    if model_name=='vgg16':
        model = torchvision.models.vgg16()
        model.classifier[-1] = torch.nn.Linear(4096, 100)
        model.load_state_dict(torch.load(directory+'vgg16_trained_67.pt'))
        fc = model.classifier
    if model_name=='vgg19':
        model = torchvision.models.vgg19()
        model.classifier[-1] = torch.nn.Linear(4096, 100)
        model.load_state_dict(torch.load(directory+'vgg19_trained_67.pt'))
        fc = model.classifier
    if model_name=='mobilenetv2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.fc = torch.nn.Linear(1280, 100)
        if not alternative:
            model.load_state_dict(torch.load(directory+'mobilenetv2_trained_57.pt'), strict=False)
        else:
            model.load_state_dict(torch.load(directory + 'alternative_mobilenetv2.pt'), strict=False)
        fc = model.classifier
    if model_name=='shufflenetv2':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = torch.nn.Linear(1024, 100)
        model.load_state_dict(torch.load(directory+'shufflenetv2_trained_54_3.pt'), strict=False)
        fc = model.fc
    if model_name=='alexnet':
        model = torchvision.models.alexnet()
        model.load_state_dict(torch.load(directory+'alexnet_trained_54_3.pt'), strict=False)
        fc = model.classifier
    if model_name=='googlenet':
        model = torchvision.models.googlenet(pretrained=True)
        model.fc = torch.nn.Linear(1024, 100)
        model.load_state_dict(torch.load(directory+'googlenet_trained_72.pt'), strict=True)
        fc = model.fc
        input_transform = model._transform_input

    return model, fc, fclayer[model_name], totallayer[model_name], input_transform
