import torch
import torchvision

# model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
# model = torchvision.models.resnet18(pretrained=True)
#
# print(model)

class PartialResNext101(torch.nn.Module):
    def __init__(self, layernum=4, pretrained=True):
        super(PartialResNext101, self).__init__()
        model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        self.layernum = layernum
        for layer in list(model.children())[:4]:
            for param in layer.parameters():
                param.grad = None
        if layernum < 5:
            self.layer1 = model.layer1
        if layernum < 6:
            self.layer2 = model.layer2
        if layernum < 7:
            self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum < 5:
            x = self.layer1(x)
        if self.layernum < 6:
            x = self.layer2(x)
        if self.layernum < 7:
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class PartialResNext50(torch.nn.Module):
    def __init__(self, layernum=4, pretrained=True):
        super(PartialResNext50, self).__init__()
        model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        self.layernum = layernum
        for layer in list(model.children())[:4]:
            for param in layer.parameters():
                param.grad = None
        if layernum < 5:
            self.layer1 = model.layer1
        if layernum < 6:
            self.layer2 = model.layer2
        if layernum < 7:
            self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum < 5:
            x = self.layer1(x)
        if self.layernum < 6:
            x = self.layer2(x)
        if self.layernum < 7:
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class PartialResNet18(torch.nn.Module):
    def __init__(self, layernum=4, pretrained=True):
        super(PartialResNet18, self).__init__()
        model = torchvision.models.resnet18(pretrained=pretrained)
        self.layernum = layernum
        for layer in list(model.children())[:4]:
            for param in layer.parameters():
                param.grad = None
        if layernum < 5:
            self.layer1 = model.layer1
        if layernum < 6:
            self.layer2 = model.layer2
        if layernum < 7:
            self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum < 5:
            x = self.layer1(x)
        if self.layernum < 6:
            x = self.layer2(x)
        if self.layernum < 7:
            x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class PartialResNet20(torch.nn.Module):
    def __init__(self, layernum=4, pretrained=True):
        super(PartialResNet20, self).__init__()
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
        self.layernum = layernum
        for layer in list(model.children())[:4]:
            for param in layer.parameters():
                param.grad = None
        if layernum < 5:
            self.layer1 = model.layer1
        if layernum < 6:
            self.layer2 = model.layer2
        if layernum < 7:
            self.layer3 = model.layer3
        # self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum < 5:
            x = self.layer1(x)
        if self.layernum < 6:
            x = self.layer2(x)
        if self.layernum < 7:
            x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class PartialResNet56(torch.nn.Module):
    def __init__(self, layernum=4, pretrained=True):
        super(PartialResNet20, self).__init__()
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
        self.layernum = layernum
        for layer in list(model.children())[:4]:
            for param in layer.parameters():
                param.grad = None
        if layernum < 5:
            self.layer1 = model.layer1
        if layernum < 6:
            self.layer2 = model.layer2
        if layernum < 7:
            self.layer3 = model.layer3
        # self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum < 5:
            x = self.layer1(x)
        if self.layernum < 6:
            x = self.layer2(x)
        if self.layernum < 7:
            x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
