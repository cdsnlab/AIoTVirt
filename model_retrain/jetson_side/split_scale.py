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
# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
# model = torchvision.models.resnext101_32x8d(pretrained=True)
model = torchvision.models.resnext50_32x4d(pretrained=True)
# model = torchvision.models.resnet18(pretrained=True)
model = model.to('cuda')
layer_num=6
front_model = torch.nn.Sequential(*(list(model.children())[:layer_num - 1]))
front_model = front_model.to('cuda')
test_loader = torch.utils.data.DataLoader(
    testset,
    # args.batch_size,
    512,
    num_workers=4,
    shuffle=False
)
# layer_num = 7


for data in test_loader:
    images, labels = data
    # images = F.interpolate(images, size=(224, 224))
    # labels = labels.to('cuda')
    # calculate outputs by running images through the network
    # outputs = model(images.to('cuda'))

    outputs = front_model(images.to('cuda'))

log = open('log.txt', 'a')
total_total_t = 0
# evens = list(range(0, len(testset)//10))
init=10
while True:
    #evens = list(range(i*1000, (i+1)*1000))
    evens = list(range(init))
    subset = torch.utils.data.Subset(testset, evens)

    test_loader = torch.utils.data.DataLoader(
        subset,
        # args.batch_size,
        1024,
        num_workers=4,
        shuffle=False
    )
    # layer_num = 7


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
            # torch.save(outputs, 'Resnet20/'+str(layer_num)+'-'+str(ind)+'.pt')
            torch.save(outputs, 'temp.pt')

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
    total_total_t += total_t
    print(init, ' %.3f' % (total_t))
    log.write(str(init) +  ' %.3f' % (total_t))
    init = int(init*1.1)
    if init>10000:
        break
# print('total time: %.3f' % (total_t), ' Time per image: %.5f' % (total_t / total))
# print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('total_total time: %.3f' % (total_total_t))
print(outputs.shape)