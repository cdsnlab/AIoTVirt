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

layer_num=4
front_model = torch.nn.Sequential(*(list(model.children())[:layer_num]))
front_model = front_model.to('cuda')
test_loader = torch.utils.data.DataLoader(
    testset,
    # args.batch_size,
    256,
    num_workers=4,
    shuffle=False
)
#for data in test_loader:
#    images, labels = data
#    outputs = front_model(images.to('cuda'))
#    break

total_total_t = 0
# evens = list(range(0, len(testset)//10))
for trial in range(1):
    total_total_t = 0
    for i in range(1):
        #evens = list(range(i*256, (i+1)*256))
        #evens = list(range(5120))
        #subset = torch.utils.data.Subset(testset, evens)

        #test_loader = torch.utils.data.DataLoader(
        #    subset,
            # args.batch_size,
        #    256,
        #    num_workers=4,
        #    shuffle=False
        #)
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
                torch.save(outputs, 'ResNext50/'+str(layer_num)+'-'+str(ind)+'.pt')
                #torch.save(outputs, 'temp.pt')

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
    print('total_total time: %.3f' % (total_total_t))
    #print(outputs.shape)