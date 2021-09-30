import torch
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import time
import copy
import torch.nn as nn
import os.path



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

criterion = nn.CrossEntropyLoss()


back_model = torchvision.models.resnext101_32x8d(pretrained=True)
# back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_repvgg_a2", pretrained=True)

# back_model = torchvision.models.resnet18(pretrained=True)
# back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
back_model = back_model.to('cuda')
version = 0
check_version =0
for batch_idx, (data, targets) in enumerate(test_loader):
    outputs = back_model(data.to('cuda'))

with torch.no_grad():
    for i in range(1000):
        start_t = time.time()
        part_time = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(test_loader):
            outputs = back_model(data.to('cuda'))


            # traintime += time.time() - start_t
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.to('cuda')).sum().item()
            total += targets.size(0)
        # part_time += traintime
        # print(predicted)
        # print(targets)

        part_time = time.time()-start_t

        print('Accuracy : %d %%' % (100 * correct / total))
        print('epoch time: %.3f' % (part_time))
        time.sleep(10)
