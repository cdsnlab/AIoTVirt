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


def split_head(
    model_name: str,
    split_point: int = 0
):
    original_model = source.__dict__[model_name](pretrained=False)
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


if __name__ == '__main__':
    model = 'resnet18'
    for i in range(model_top_layers[model]):
        m = split_head(model, i)
        print(i)
        print(m)
