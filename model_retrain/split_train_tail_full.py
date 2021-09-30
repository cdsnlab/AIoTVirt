import torch
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import time
import copy
import torch.nn as nn

trainset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    trainset,
    # args.batch_size,
    256,
    num_workers=4,
    shuffle=False
)
criterion = nn.CrossEntropyLoss()

# back_model = torchvision.models.resnext101_32x8d(pretrained=True)
# back_model = torchvision.models.resnet18(pretrained=True)
back_model = torchvision.models.resnext50_32x4d(pretrained=True)

# back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)

print(back_model)
# back_model = torchvision.models.resnet18(pretrained=True)
# back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)

back_model = back_model.to('cuda')
optimizer = torch.optim.SGD(back_model.parameters(), lr=0.001, momentum=0.9)
version = 0
trigger_time = time.time()
outputs_old_list = []
while (True):
    part_time = 0
    correct = 0
    total = 0
    start_t = time.time()
    if time.time() > trigger_time:
        trigger_time = time.time() + 1
        print('triggered')
        for epoch in range(1):
            for batch_idx, (data, targets) in enumerate(train_loader):
                outputs = back_model(data.to('cuda'))

                losses = criterion(outputs, targets.to('cuda'))
                if version > 0:
                    outputs_old = outputs_old_list[batch_idx]
                    g = torch.sigmoid(outputs)
                    with torch.no_grad():
                        q_i = torch.sigmoid(outputs_old)

                    losses += 0.1 * torch.nn.functional.binary_cross_entropy(g[:, :100], q_i[:, :100])
                    outputs_old_list[batch_idx] = outputs
                else:
                    outputs_old_list.append(outputs)
                optimizer.zero_grad()

                losses.backward()

                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets.to('cuda')).sum().item()
                total += targets.size(0)
        torch.save(back_model.state_dict(), 'back_model.pt')
        version += 1
        torch.save({'version': (version)}, 'version.pt')

        part_time = time.time() - start_t

        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print('epoch time: %.3f' % (part_time))
    else:
        continue
