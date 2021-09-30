import torchvision
from torchvision import datasets
import torch
import utils_func

def infer(x):
    return 1


class Icarl:
    def __init__(self, size='small', class_num = 10):
        self.exemplar = []
        self.infer = infer
        self.memory_size = 1024
        self.class_num = class_num
        self.size = size
        self.heads = torch.nn.ModuleList()
        self.task_cls = []
        self.task_offset = []

        if size=='small':
            self.model = torchvision.models.resnet50(pretrained=True)
            # self.model = torchvision.models.resnet18(pretrained=False)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.class_num, bias=False)
            self.model = self.model.to('cuda')

            ct = 0
            for child in self.model.children():
                ct += 1
                print(ct)
                # print(child, child[0].parameters())
                if ct < 7:
                    for param in child.parameters():
                        param.requires_grad = False
            testset = datasets.CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor()
            )

            test_loader = torch.utils.data.DataLoader(
                testset,
                # args.batch_size,
                64,
                num_workers=4,
                #shuffle=True
            )

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            import torch.nn.functional as F
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    # images = F.interpolate(images, size=(224, 224))
                    labels = labels.to('cuda')
                    # calculate outputs by running images through the network
                    outputs = self.model(images.to('cuda'))
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # print(predicted, labels)
                    # break
                    correct += (predicted == labels).sum().item()
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))



        elif size=='pretrained':

            self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
            self.model = self.model.to('cuda')
            ct = 0
            for child in self.model.children():
                ct += 1
                print(ct)
                # print(child, child[0].parameters())
                if ct < 1: #3
                    for param in child.parameters():
                        param.requires_grad = False

            testset = datasets.CIFAR100(
                root="data",
                train=False,
                download=True,
                transform=torchvision.transforms.ToTensor()
            )
            human_testset = utils_func.DatasetMaker(
                [utils_func.get_class_i(testset.data, testset.targets, i) for i in [2, 11, 35, 46, 98]],
                torchvision.transforms.ToTensor(),
                [2, 11, 35, 46, 98]
            )

            test_loader = torch.utils.data.DataLoader(
                testset,
                # args.batch_size,
                64,
                num_workers=4,
                #shuffle=True
            )
            human_test_loader = torch.utils.data.DataLoader(
                human_testset,
                # args.batch_size,
                64,
                num_workers=4,
                # shuffle=True
            )

            correct = 0
            total = 0
            human_correct = 0
            human_total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            import torch.nn.functional as F
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    # images = F.interpolate(images, size=(224, 224))
                    labels = labels.to('cuda')
                    # calculate outputs by running images through the network
                    outputs = self.model(images.to('cuda'))
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # print(predicted, labels)
                    # break
                    correct += (predicted == labels).sum().item()
                for data in human_test_loader:
                    images, labels = data
                    # images = F.interpolate(images, size=(224, 224))
                    labels = labels.to('cuda')
                    # calculate outputs by running images through the network
                    outputs = self.model(images.to('cuda'))
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    human_total += labels.size(0)
                    # print(predicted, labels)
                    # break
                    human_correct += (predicted == labels).sum().item()


            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            print('Accuracy of person images: %d %%' % (100 * human_correct / human_total), human_correct, human_total)



        else:
            self.model = torchvision.models.resnext101_32x8d(pretrained=True)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.class_num, bias=False)


    def classifier(self, x):
        mu = [0]*len(self.exemplar)
        for i in range(len(self.exemplar)):
            exemplar_len = len(self.exemplar[i])
            for j in range(exemplar_len):
                mu[i] += self.infer(self.exemplar[j])/exemplar_len
        phi = self.infer(x)
        y_star = abs(mu[0]-phi)
        index = 0
        for i in range(len(mu)):
            if y_star>mu[i]:
                index = i
                y_star = mu[i]
        return index

    def incre_train(self, images):
        self.update_rep()
        total_len = len(images) + len(self.exemplar)
        m = self.memory_size//total_len

        self.reduce_exemplar(m)
        self.update_rep(m)

    def update_rep(self):
        pass

    def reduce_exemplar(self, m):
        pass

    def construct_exemplar(self, m):
        pass
