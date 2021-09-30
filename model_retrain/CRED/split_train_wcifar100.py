import torch
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import time
import copy
import torch.nn as nn
from mydataset import MyDataset

# trainset = datasets.CIFAR100(
#    root="data",
#    train=True,
#    download=True,
#    transform=torchvision.transforms.ToTensor()
# )
mydataset = MyDataset(csv_file='../files.csv', root_dir='../images/', label_file='../label.csv', train=True)

evens = list(range(0, len(mydataset)//10*3, 1))
mydataset = torch.utils.data.Subset(mydataset, evens)


train_loader = torch.utils.data.DataLoader(
    #    trainset,
    mydataset,
    # args.batch_size,
    64,
    num_workers=4,
    shuffle=False
)

mydataset = MyDataset(csv_file='../files.csv', root_dir='../images/', label_file='../label.csv', train=False)


#evens = list(range(0, len(mydataset), 10))
#mydataset = torch.utils.data.Subset(mydataset, evens)


test_loader = torch.utils.data.DataLoader(
    #    trainset,
    mydataset,
    # args.batch_size,
    64,
    num_workers=4,
    shuffle=False
)

dataset_source = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

#evens = list(range(0, len(dataset_source), 10))
#dataset_source = torch.utils.data.Subset(dataset_source, evens)


cifar_train_loader = torch.utils.data.DataLoader(
    dataset_source,
    64,
    num_workers=4,
    shuffle=True
)
dataset_source = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
cifar_test_loader = torch.utils.data.DataLoader(
    dataset_source,
    64,
    num_workers=4,
    shuffle=True
)

criterion = nn.CrossEntropyLoss()

# back_model = torchvision.models.resnext101_32x8d(pretrained=True)
# back_model = torchvision.models.resnet18(pretrained=True)
#back_model = torchvision.models.resnext50_32x4d(pretrained=True)
#back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_shufflenetv2_x1_0", pretrained=True)

back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)

layer_stack = 0
print(back_model)
for name, parameter in back_model.named_parameters():
    if(name[len(name)-6:]=='weight'):
        layer_stack+=1
print(layer_stack)
# back_model = torchvision.models.resnet18(pretrained=True)
#back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)

back_model = back_model.to('cuda')
optimizer = torch.optim.SGD(back_model.parameters(), lr=0.001, momentum=0.9)
version = 0
trigger_time = time.time()
outputs_old_list = []
minor_first = True
import random

while (True):
    part_time = 0
    correct = 0
    total = 0
    start_t = time.time()
    train_data_loader = cifar_train_loader
    minor_first = True
    if minor_first:
        train_data_loader = train_loader

    if time.time() > trigger_time:
        trigger_time = time.time() + 1
        print('triggered')
        for batch_idx, (data, targets) in enumerate(train_data_loader):
            if minor_first:
                if random.random() < 0:
                    continue
            else:
                if random.random() < 0:
                    continue

            outputs = back_model(data.to('cuda'))
            losses = criterion(outputs, targets.to('cuda'))

            # if version > 0:
            #     outputs_old = outputs_old_list[batch_idx]
            #     g = torch.sigmoid(outputs)
            #     with torch.no_grad():
            #         q_i = torch.sigmoid(outputs_old)
            #
            #     losses += 0.1 * torch.nn.functional.binary_cross_entropy(g[:, :100], q_i[:, :100])
            #     outputs_old_list[batch_idx] = outputs
            # else:
            #     outputs_old_list.append(outputs)
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
        correct = 0
        top5_correct = 0
        total = 0
        top5_humans = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in cifar_test_loader:
                images, labels = data
                # images = F.interpolate(images, size=(224, 224))
                labels = labels.to('cuda')
                # calculate outputs by running images through the network
                outputs = back_model(images.to('cuda'))
                # the class with the highest energy is what we choose as prediction
                # print(labels.shape)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # print(predicted, labels)
                # break
                correct += (predicted == labels).sum().item()
            human_correct = 0
            human_total = 0
            for data in test_loader:
                images, labels = data
                # images = F.interpolate(images, size=(224, 224))
                labels = labels.to('cuda')
                # calculate outputs by running images through the network
                outputs = back_model(images.to('cuda'))
                # the class with the highest energy is what we choose as prediction
                # print(labels)

                _, predicted = torch.max(outputs.data, 1)
                human_total += labels.size(0)
                # print(predicted, labels)
                # break
                human_correct += (predicted == labels).sum().item()
            print('Top1 Accuracy: %d %%' % (100 * correct / total))
            print('Top1 Accuracy of person images: %d %% ' % (100 * human_correct / human_total))
            if (correct / total) < (human_correct / human_total):
                minor_first = False
                print('Train all')
            else:
                minor_first = True
                print('Train human only')

    else:
        continue
