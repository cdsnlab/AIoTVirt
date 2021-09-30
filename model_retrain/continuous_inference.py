import torch
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import time
import copy
import os.path
from partial_model import PartialResNext101, PartialResNet18, PartialResNet20, PartialResNext50




testset = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_loader = torch.utils.data.DataLoader(
    testset,
    # args.batch_size,
    256,
    num_workers=4,
    shuffle=False
)

import torch.nn as nn

criterion = nn.CrossEntropyLoss()

inters = []

layercut = 6

for i in range(40):
    # inters.append(torch.load('../ResNet18_weights/t' + str(layercut) + '-' + str(i + 1) + '.pt').to('cpu'))
    inters.append(torch.load('../ResNext50_weights/t' + str(layercut) + '-' + str(i + 1) + '.pt').to('cpu'))
    # inters.append(torch.load('../Resnet20_weights/t' + str(layercut) + '-' + str(i + 1) + '.pt').to('cpu'))


# back_model = PartialResNext101(layercut)
# back_model = PartialResNet20(layercut)
back_model = PartialResNext50(layercut)

torch.save(back_model.state_dict(), 'back_model.pt')
back_model = back_model.to('cuda')
version = 0
check_version =0

for batch_idx, (data, targets) in enumerate(test_loader):
    data = inters[batch_idx].to('cuda')
    outputs = back_model(data)


with torch.no_grad():
    for i in range(1000):
        start_t = time.time()
        if os.path.isfile('version.pt'):
            check_version = torch.load('version.pt')['version']
        if check_version>version:
            version = check_version
            back_model.load_state_dict(torch.load('back_model.pt'))
        part_time = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(test_loader):
            data = inters[batch_idx].to('cuda')

            outputs = back_model(data)


            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.to('cuda')).sum().item()
            total += targets.size(0)

        part_time = time.time()-start_t

        print('Accuracy : %d %%' % (100 * correct / total))
        print('epoch time: %.3f' % (part_time))

