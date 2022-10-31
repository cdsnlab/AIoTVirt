import os
import sys
import torch
import copy

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
sys.path.append(BASE_DIR)

try:
    from models import source
except ModuleNotFoundError:
    raise

model_top_layers = {
    'mobilenetv2': 21,
    'efficientnet_b0': 9,
    'googlenet': 21,
    'resnet18': 13
}
fclayer = {'resnet18': 1, 'resnet34': 1, 'resnet50': 1, 'resnet101': 1,
           'resnext50': 1, 'resnext101': 1,
           'vgg11': 7, 'vgg13': 7, 'vgg16': 7, 'vgg19': 7,
           'mobilenetv2': 2, 'shufflenetv2': 1, 'alexnet': 7, 'googlenet': 1}
totallayer = {'resnet18': 14, 'resnet34': 22, 'resnet50': 22, 'resnet101': 39,
              'resnext50': 22, 'resnext101': 39,
              'vgg11': 29, 'vgg13': 33, 'vgg16': 39, 'vgg19': 45,
              'mobilenetv2': 21, 'shufflenetv2': 24, 'alexnet': 21,
              'googlenet': 18}


def split_head(
    model_name: str,
    split_point: int = 0,
    pretrained_weights=None
):
    original_model = source.__dict__[model_name](pretrained=False)
    if pretrained_weights is not None:
        state_dict = torch.load(pretrained_weights)
        original_model.load_state_dict(state_dict)
    if model_name == 'efficientnet_0':
        torch.nn.init.xavier_uniform_(original_model, gain=1.0)
    top_layer = model_top_layers[model_name]
    modulelist = []

    ct = 0
    for child in original_model.children():
        if isinstance(child, torch.nn.Sequential):
            for sub_child in range(len(child)):
                ct += 1
                if ct <= split_point:
                    modulelist.append(child[sub_child])
        else:
            ct += 1
            if ct <= split_point:
                modulelist.append(child)
    if split_point == top_layer:
        model = copy.deepcopy(original_model)
    else:
        model = torch.nn.Sequential(*modulelist)

    del original_model

    return model


def pre_fc(x, name, train=False):
    if name == 'shufflenetv2':
        x = x.mean([2, 3])
    else:
        if name == 'mobilenetv2':
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
    return x


class TailModel(torch.nn.Module):
    def __init__(self, model_name, split_point, pretrained_weights=None):
        super(TailModel, self).__init__()

        self.original_model = source.__dict__[model_name](pretrained=False)
        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights)
            self.original_model.load_state_dict(state_dict)
        self.model = None
        self.model_name = model_name
        self.split_point = totallayer[self.model_name] - fclayer[self.model_name] + \
            1 - split_point
        self.fclayer = fclayer[model_name]
        self.totallayer = totallayer[model_name]
        self.defactolayer = self.totallayer - self.fclayer + 1
        self.modulelist = []

        ct = 0
        for child in self.original_model.children():
            child = copy.deepcopy(child)
            if isinstance(child, torch.nn.Sequential):
                for sub_child in range(len(child)):
                    ct += 1
                    if ct + self.split_point > self.defactolayer:
                        self.modulelist.append(copy.deepcopy(child[sub_child]))
            else:
                ct += 1
                if ct + self.split_point > self.defactolayer:
                    self.modulelist.append(child)
        self.fc = copy.deepcopy(self.original_model.fc)
        if self.split_point == 1:
            self.model = self.fc
        else:
            self.modulelist = self.modulelist[:-self.fclayer]
            self.model = torch.nn.Sequential(*self.modulelist)
        del self.original_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.split_point == 0:
            return x
        if self.split_point == 1:
            x = pre_fc(x, self.model_name)
            x = self.model(x)
        else:
            x = self.model(x)
            x = pre_fc(x, self.model_name)
            x = self.fc(x)
        return x


if __name__ == '__main__':
    model = 'resnet18'
    for i in range(model_top_layers[model]):
        m = split_head(model, i)
        print(i)
        print(m)
