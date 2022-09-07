import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


def mobilenetv2_split_head(split_point=0):
    original_model = torchvision.models.mobilenet_v2(pretrained=False)
    defactolayer = 21
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
    if split_point == defactolayer:
        model = copy.deepcopy(original_model)
    else:
        model = torch.nn.Sequential(*modulelist)
    
    del original_model
    
    return model


if __name__ == '__main__':
    for i in range(22):
        m = mobilenetv2_split_head(i)
        print(i)
        print(m)
