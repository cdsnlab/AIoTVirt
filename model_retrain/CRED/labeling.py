import torch
from torchvision import datasets
import torchvision
import torch.nn.functional as F
import time
import copy
import torch.nn as nn
from mydataset import MyDataset


trainset = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
mydataset = MyDataset(csv_file='../files.csv', root_dir='../images/')

train_loader = torch.utils.data.DataLoader(
#    trainset,
    mydataset,
    # args.batch_size,
    64,
    num_workers=4,
    shuffle=False
)
f = open("../label.csv",'w')
f.write("label\n")
#back_model = torchvision.models.resnext101_32x8d(pretrained=True)
back_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_repvgg_a2", pretrained=True)
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
        #print('triggered')
        for epoch in range(1):
            for batch_idx, (data, targets) in enumerate(train_loader):
                outputs = back_model(data.to('cuda'))
                _, predicted = torch.max(outputs.data, 1)
                #print(predicted.shape)
                for i, x in enumerate(predicted) :
                    #f.write(str(x.item())+'\n')
                    f.write(str(46)+'\n')

        part_time = time.time() - start_t

        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        print('epoch time: %.3f' % (part_time))
        break
    else:
        continue
f.close()