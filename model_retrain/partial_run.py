# import torch
# from torchvision import datasets
# import torchvision
# import torch.nn.functional as F
# import time
# from partial_model import PartialResNext101, PartialResNet18, PartialResNet20, PartialResNext50
#
#
# testset = datasets.CIFAR100(
#     root="data",
#     train=True,
#     download=True,
#     transform=torchvision.transforms.ToTensor()
# )
#
# # evens = list(range(0, len(testset)//10))
# # evens = list(range(1000))
# # testset = torch.utils.data.Subset(testset, evens)
#
#
# test_loader = torch.utils.data.DataLoader(
#     testset,
#     # args.batch_size,
#     256,
#     num_workers=4,
#     shuffle=False
# )
#
# # model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
# model = torchvision.models.resnext101_32x8d(pretrained=True)
# # model = torchvision.models.resnet18(pretrained=True)
# model = model.to('cuda')
# # print(model)
#
# # inters = []
# # for i in range(196):
# #     inters.append(torch.load('../ResNext_weights/' + str(layercut) + '-' + str(i + 1) + '.pt').to('cpu'))
#
#
# # layer_num = 7
#
# for layer_num in [4, 4, 5, 6]:
#     inters = []
#     for i in range(196):
#         inters.append(torch.load('../ResNext_weights/' + str(layer_num) + '-' + str(i + 1) + '.pt').to('cpu'))
#
#     if layer_num == 0:
#         layer_num = 1
#     # front_model = torch.nn.Sequential(*(list(model.children())[:layer_num - 1]))
#     # front_model = front_model.to('cuda')
#     # back_model = torch.nn.Sequential(*(list(model.children())[layer_num:7]))
#     # back_model = back_model.to('cuda')
#     # class_model = torch.nn.Sequential(*(list(model.children())[7:])).to('cuda')
#     # print(front_model, back_model)
#
#     back_model = PartialResNext101(layer_num)
#     back_model = back_model.to('cuda')
#
#     correct = 0
#     total = 0
#
#     start_t = time.time()
#     total_t = 0
#     with torch.no_grad():
#         ind = 0
#         for batch_idx, (data, targets) in enumerate(test_loader):
#             ind += 1
#             # images, labels = data
#             # images = F.interpolate(images, size=(224, 224))
#             # labels = labels.to('cuda')
#             # calculate outputs by running images through the network
#             outputs = model(data.to('cuda'))
#
#             # outputs = front_model(images.to('cuda'))
#             # torch.save(outputs, 'Resnet20/'+str(layer_num)+'-'+str(ind)+'.pt')
#             # torch.save(outputs, 'temp.pt')
#             # data = inters[batch_idx].to('cuda')
#             #
#             # outputs = back_model(data)
#
#             # the class with the highest energy is what we choose as prediction
#             # _, predicted = torch.max(outputs.data, 1)
#             total += targets.size(0)
#             # print(predicted, labels)
#             # break
#             # correct += (predicted == labels).sum().item()
#     total_t = time.time() - start_t
#     print('total time: %.3f' % (total_t), ' Time per image: %.5f' % (total_t / total))
#     print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
#     print(outputs.shape)

import torch
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import time

testset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# evens = list(range(0, len(testset)//10))
# evens = list(range(1000))
# testset = torch.utils.data.Subset(testset, evens)


test_loader = torch.utils.data.DataLoader(
    testset,
    # args.batch_size,
    256,
    num_workers=4,
    shuffle=False
)

# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
model = torchvision.models.resnext101_32x8d(pretrained=True)
# model = torchvision.models.resnet18(pretrained=True)
model = model.to('cuda')
# print(model)

# layer_num = 7

for layer_num in [4, 4, 5, 6]:
    if layer_num == 0:
        layer_num = 1
    front_model = torch.nn.Sequential(*(list(model.children())[:layer_num - 1]))
    front_model = front_model.to('cuda')
    # back_model = torch.nn.Sequential(*(list(model.children())[layer_num:7]))
    # back_model = back_model.to('cuda')
    # class_model = torch.nn.Sequential(*(list(model.children())[7:])).to('cuda')
    # print(front_model, back_model)

    correct = 0
    total = 0

    start_t = time.time()
    total_t = 0
    with torch.no_grad():
        ind = 0
        for data in test_loader:
            ind += 1
            images, labels = data
            # images = F.interpolate(images, size=(224, 224))
            # labels = labels.to('cuda')
            # calculate outputs by running images through the network
            # outputs = model(images.to('cuda'))

            outputs = front_model(images.to('cuda'))
            torch.save(outputs, 'Resnet20/'+str(layer_num)+'-'+str(ind)+'.pt')
            # torch.save(outputs, 'temp_tensor.pt')

            # outputs = back_model(outputs)
            # outputs = outputs.view(outputs.size(0), -1)
            # outputs = class_model(outputs)

            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted, labels)
            # break
            # correct += (predicted == labels).sum().item()
    total_t = time.time() - start_t
    print('total time: %.3f' % (total_t), ' Time per image: %.5f' % (total_t / total))
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print(outputs.shape)