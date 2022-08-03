
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import collections
import sys
from time import time

from utils import toGreen, toRed, progress_bar
from load_partial_model import model_spec


'''
The information of layer in each model is listed in dictionary.
These are used when splitting the model.
'''
fclayer = {'resnet18': 1, 'resnet34': 1, 'resnet50': 1, 'resnet101': 1,
            'resnext50': 1, 'resnext101': 1,
            'vgg11': 7, 'vgg13': 7, 'vgg16': 7, 'vgg19': 7,
            'mobilenetv2': 2, 'shufflenetv2': 1, 'alexnet': 7, 'googlenet': 1}
totallayer = {'resnet18': 14, 'resnet34': 22, 'resnet50': 22, 'resnet101': 39,
            'resnext50': 22, 'resnext101': 39,
            'vgg11': 29, 'vgg13': 33, 'vgg16': 39, 'vgg19': 45,
            'mobilenetv2': 21, 'shufflenetv2': 24, 'alexnet': 21, 'googlenet': 18}

'''
To make the right size of tensor before fc layer
'''
def pre_fc(x, name, train=False):
    if name == 'shufflenetv2':
        x = x.mean([2, 3])
    else:
        if name == 'mobilenetv2':
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
    return x

'''
When new class is added into last layer, that output node should be initilized.
In the LwF paper. the authors said xavier initialization is used.
'''
def xavier_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, nonlinearity='sigmoid')

'''
The Trainer consists of Head model and Tail model.
This can excute the forward propagation of each model, and makes them learn incrementally.
The Knowledge Distillation(KD) method is used in Lwf, so the old model(teacher model) should exist.
'''
class Trainer():
    def __init__(self, config):
        self.config = config
        model, _, _, _, _ = model_spec(self.config.network, False)

        # self.model = copy.deepcopy(model)
        self.device = self.config.device
        self.model = model.to(self.device)
        self.alpha = self.config.alpha
        self.T = self.config.temperature
        self.old_head_model = None
        self.old_tail_model = None
        '''
        The head model can be defined in other device.
        '''
        self.head_model = None
        self.tail_model = None
        self.input_transform = None
        self.tail_optimizer = None
        self.split_point = None
        self.loss_function = nn.CrossEntropyLoss()
        self.inference_latency = dict()
        self.retrain_time = dict()
        self.network_latency = dict()
        
        if self.config.network == 'googlenet':
            self.input_transform = self.model._transform_input
            
    '''
    Measure the whold latencies on all split point settings.
    1. forward propagation time
    2. retrain time
    3. network latency
    '''
    def measure_latency(self, dataloader):
        for split_point in range(totallayer[self.config.network]):
            head_model = HeadModel(split_point).to(self.device)
            tail_model = TailModel(totallayer[self.config.network] - fclayer[self.config.network] + 1 - split_point).to(self.device)
            tail_optimizer = torch.optim.SGD(tail_model.parameters(), lr=0.001, momentum=0.9)
            for batch_idx, data in enumerate(dataloader):
                t = time()
                images, targets = data
                if self.config.network == 'alexnet' or self.config.network == 'googlenet':
                    intermediate_tensor = head_model.forward(torch.nn.functional.interpolate(images.to(self.device), size=(64, 64)))
                else:
                    intermediate_tensor = head_model.forward(images.to(self.device))
                self.network_latency[split_point] = sys.getsizeof(intermediate_tensor)
                outputs = tail_model.forward(intermediate_tensor)
                self.inference_latency[split_point] = time()-t
                loss = self.loss_function(outputs, targets)
                loss.backward()
                tail_optimizer.zero_grad()
                tail_optimizer.step()
                self.retrain_time[split_point] = time() - t
                break
                
        
    '''
    Split the model based on split_point and define the optimizer again.
    '''
    def set_network(self, split_point):
        self.split_point = split_point
        self.head_model = HeadModel(split_point).to(self.device)
        self.tail_model = TailModel(totallayer[self.config.network] - fclayer[self.config.network] + 1 - split_point).to(self.device)
        self.tail_optimizer = torch.optim.SGD(self.tail_model.parameters(), lr=0.001, momentum=0.9)
        
    '''
    First of all, head model outputs the intermediate tensor, 
    and the tail model receives that and make output.
    '''
    def forward(self, x: torch.Tensor, is_train = False) -> torch.Tensor:
        if not is_train:
            with torch.no_grad():
                output = self.tail_model.forward(self.head_model.forward(x))
        else:
            output = self.tail_model.forward(self.head_model.forward(x))
            
        return output
    
    '''
    Old model version of forward function.
    '''
    def old_forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.old_tail_model.forward(self.old_head_model.forward(x))
        return output
    
    '''
    Return computation latency at the split point setting.
    The values are already calcultated in set_retrain_time(self, dataloader)
    '''
    def get_time_train_one_step(self, split_point):
        return self.retrain_time[split_point]
        
    '''
    Return all retrain computation latency at the split point setting.
    The values are already calcultated in set_retrain_time(self, dataloader)
    '''
    def get_time_all_train(self):
        return self.retrain_time
    
    '''
    Return all network latency at the split point setting.
    The values are already calcultated in set_retrain_time(self, dataloader)
    '''
    def get_time_all_network_latency(self):
        return self.network_latency
    
    '''
    Return inference computation latency at the split point setting.
    The values are already calcultated in set_retrain_time(self, dataloader)
    '''
    def get_time_all_forward_prop(self):
        return self.inference_latency
        
    '''
    Migrate the weight and bias in original fc layer to new fc layer.
    And if new class is detected, then add new class into last layer.
    And then retrain the tail model in IL manner.
    You can use this function when measuring retrain one epoch time by setting the epoch param to 1.
    '''
    def incremental_learning(self, dataloader, epoch, num_new_class, is_profile=False):
        # migrate the head_model and tail model
        if not is_profile:
            self.old_head_model, self.old_tail_model = copy.deepcopy(self.head_model), copy.deepcopy(self.tail_model)
            self.old_head_model.eval()
            self.old_tail_model.eval()
        
        # add new class
        self.tail_model = self.tail_model.to('cpu')
        # Old number of input/output channel of the last FC layer in old model
        in_features = self.tail_model.classifier[6].in_features
        out_features = self.tail_model.classifier[6].out_features
        # Old weight/bias of the last FC layer
        weight = self.tail_model.classifier[6].weight.data
        bias = self.tail_model.classifier[6].bias.data
        # New number of output channel of the last FC layer in new model
        new_out_features = num_new_class+out_features
        # Creat a new FC layer and initial it's weight/bias
        new_fc = nn.Linear(in_features, new_out_features)
        xavier_normal_init(new_fc.weight)
        new_fc.weight.data[:out_features] = weight
        new_fc.bias.data[:out_features] = bias
        # Replace the old FC layer
        self.tail_model.classifier[6] = new_fc
        # CUDA
        self.tail_model = self.tail_model.to(self.device)
        
        # train
        self.head_model.eval()
        self.tail_model.train()
                
        t = time()
        correct, total = 0, 0
        for e in range(epoch):
            correct_one_epoch, total_one_epoch = 0, 0
            for batch_idx, data in enumerate(dataloader):
                images, targets = data
                if self.config.network == 'alexnet' or self.config.network == 'googlenet':
                    outputs = self.forward(
                        torch.nn.functional.interpolate(images.to(self.device), size=(64, 64)), is_train=True)
                    soft_target = self.old_forward(
                        torch.nn.functional.interpolate(images.to(self.device), size=(64, 64)), is_train=True)
                else:
                    outputs = self.forward(images.to(self.device))
                    soft_target = self.old_forward(images.to(self.device))

                loss1 = self.loss_function(outputs, targets)
                outputs_S = F.softmax(outputs[:,:out_features]/self.T,dim=1)
                outputs_T = F.softmax(soft_target[:,:out_features]/self.T,dim=1)

                loss2 = outputs_T.mul(-1*torch.log(outputs_S))
                loss2 = loss2.sum(1)
                loss2 = loss2.mean()*self.T*self.T
                loss = loss1*self.alpha+loss2*(1-self.alpha)
                loss.backward(retain_graph=True)
                self.tail_optimizer.zero_grad()
                self.tail_optimizer.step()
        
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets.to(self.device)).sum().item()
                total += targets.size(0)
                correct_one_epoch += (predicted == targets.to(self.device)).sum().item()
                total_one_epoch += targets.size(0)
                
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (loss.item()/(batch_idx+1), 100.*correct_one_epoch/total_one_epoch, correct_one_epoch, total_one_epoch))
            print(toGreen('Model: {} Split Point: {} Retrain Accuracy: {}'.format(self.config.network, self.split_point, correct/total)))
        
        IL_time = time() - t
        self.head_model.eval()    
        self.tail_model.eval()
        return correct/total, IL_time
    
    '''
    test the model.
    '''
    def test(self, dataloader):
        correct, total = 0, 0
        
        # eval mode
        self.head_model.eval()
        self.tail_model.eval()
                
        for batch_idx, data in enumerate(dataloader):
            images, targets = data
            if self.config.network == 'alexnet' or self.config.network == 'googlenet':
                outputs = self.forward(torch.nn.functional.interpolate(images.to(self.device), size=(64, 64)))
            else:
                outputs = self.forward(images.to(self.device))
    
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets.to(self.device)).sum().item()
            total += targets.size(0)
            correct_one_epoch += (predicted == targets.to(self.device)).sum().item()
            total_one_epoch += targets.size(0)
            
        print(toGreen('Model: {} Split Point: {} Retrain Accuracy: {}'.
                        format(self.config.network, self.split_point, correct/total)))
            
        return correct/total
                
                
    '''
    HeadModel class is defined in Trainer class to access the variables in Trainer conveniently.
    The HeadModel is front partial model in Trainer. -> (input): data, (output): intermediate tensor.
    It can be decided via the layernum when make instance of HeadModel.
    '''
    class HeadModel(nn.Module):
        def __init__(self, layernum):
            super(HeadModel, self).__init__()
            # self.name = name
            # model, fc ,fclayer, totallayer, input_transform = model_spec(name, alternative)
            
            self.model = None
            self.layernum = layernum
            self.fclayer = fclayer[Trainer.config.network]
            self.totallayer = totallayer[Trainer.config.network]
            self.defactolayer = self.totallayer - self.fclayer + 1
            self.modulelist = []
            self.input_transform = Trainer.input_transform
            
            ct = 0
            for child in Trainer.model.children():
                if isinstance(child, torch.nn.Sequential):
                    for sub_child in range(len(child)):
                        ct += 1
                        if ct <= self.layernum:
                            self.modulelist.append(child[sub_child])
                else:
                    ct += 1
                    if ct <= self.layernum:
                        self.modulelist.append(child)
            if self.layernum == self.defactolayer:
                self.model = Trainer.model
            else:
                self.model = torch.nn.Sequential(*self.modulelist)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.input_transform is not None and self.layernum < self.defactolayer:
                x = self.input_transform(x)

            if self.layernum == 0:
                return x

            x = self.model(x)
            return x
        
        
    '''
    TailModel class is defined in Trainer class to access the variables in Trainer conveniently.
    The TailModel is front partial model in Trainer. -> (input): intermediate tensor, (output): logit value.
    It can be decided via the layernum when make instance of TailModel.
    Retraining is only conducted in TailModel.
    '''
    class TailModel(nn.Module):
        def __init__(self, layernum):
            super(TailModel, self).__init__()
            # model, fc ,fclayer, totallayer, _ = model_spec(name, False)

            self.model = None
            self.layernum = layernum
            self.fclayer = fclayer[Trainer.config.network]
            self.totallayer = totallayer[Trainer.config.network]
            self.defactolayer = self.totallayer - self.fclayer + 1
            self.modulelist = []

            ct = 0
            for child in Trainer.model.children():
                child = copy.deepcopy(child)
                if isinstance(child, torch.nn.Sequential):
                    for sub_child in range(len(child)):
                        ct += 1
                        if ct + self.layernum > self.defactolayer:
                            self.modulelist.append(copy.deepcopy(child[sub_child]))
                else:
                    ct += 1
                    if ct + self.layernum > self.defactolayer:
                        self.modulelist.append(child)
            self.fc = copy.deepcopy(Trainer.model.fc)
            if self.layernum == 1:
                self.model = self.fc
            else:
                self.modulelist = self.modulelist[:-self.fclayer]
                self.model = torch.nn.Sequential(*self.modulelist)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.layernum == 0:
                return x
            if self.layernum == 1:
                x = pre_fc(x, Trainer.config.network)
                x = self.model(x)
            else:
                x = self.model(x)
                x = pre_fc(x, Trainer.config.network)
                x = self.fc(x)
            return x
        
    
    
