
from posixpath import split
from termios import CEOL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import collections
import sys

# from time import time
import time
from torch.utils.tensorboard import SummaryWriter
from utils import toGreen, toRed, toYellow, toBlue, toCyan, progress_bar
from load_partial_model import model_spec
from torch.autograd import Variable
from torchsummary import summary

INIT_LR = 1e-6
print('initial lr {}'.format(INIT_LR))

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

def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    if base_lr > 1e-4:
        new_lr = base_lr * factor
        # print('Set lr to ', new_lr)
    else:
        new_lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return optimizer

'''
The HeadModel is front partial model in Trainer. -> (input): data, (output): intermediate tensor.
It can be decided via the layernum when make instance of HeadModel.
'''
class HeadModel(nn.Module):
    def __init__(self, name, layernum, original_model):
        super().__init__()
        # self.name = name
        # model, fc ,fclayer, totallayer, input_transform = model_spec(name, alternative)
        
        self.original_model = copy.deepcopy(original_model)
        self.model = None
        self.name = name
        self.layernum = layernum
        self.fclayer = fclayer[name]
        self.totallayer = totallayer[name]
        self.defactolayer = self.totallayer - self.fclayer + 1
        self.modulelist = []
        
        ct = 0
        for child in self.original_model.children():
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
            self.model = self.original_model
        else:
            self.model = torch.nn.Sequential(*self.modulelist)
        
        del self.original_model
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum == 0:
            return x

        x = self.model(x)
        return x
    
    
'''
The TailModel is front partial model in Trainer. -> (input): intermediate tensor, (output): logit value.
It can be decided via the layernum when make instance of TailModel.
Retraining is only conducted in TailModel.
'''
class TailModel(nn.Module):
    def __init__(self, name, layernum, original_model):
        super(TailModel, self).__init__()
        # model, fc ,fclayer, totallayer, _ = model_spec(name, False)

        self.original_model = copy.deepcopy(original_model)
        self.model = None
        self.name = name
        self.layernum = layernum
        self.fclayer = fclayer[name]
        self.totallayer = totallayer[name]
        self.defactolayer = self.totallayer - self.fclayer + 1
        self.modulelist = []

        ct = 0
        for child in self.original_model.children():
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
        self.fc = copy.deepcopy(self.original_model.fc)
        if self.layernum == 1:
            self.model = self.fc
        else:
            self.modulelist = self.modulelist[:-self.fclayer]
            self.model = torch.nn.Sequential(*self.modulelist)
        del self.original_model
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layernum == 0:
            return x
        if self.layernum == 1:
            x = pre_fc(x, self.name)
            x = self.model(x)
        else:
            x = self.model(x)
            x = pre_fc(x, self.name)
            x = self.fc(x)
        return x


'''
The Trainer consists of Head model and Tail model.
This can excute the forward propagation of each model, and makes them learn incrementally.
The Knowledge Distillation(KD) method is used in Lwf, so the old model(teacher model) should exist.
'''
class Trainer():
    def __init__(self, config):
        self.config = config
        self.name = self.config.network
        self.dataset = self.config.dataset
        model, fc, _, _, _ = model_spec(self.name, self.dataset)
        model.load_state_dict(torch.load('./ckpt/pretrain/' + self.name + '_' + self.dataset + '.pt', map_location = 'cpu'))
        
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
        self.output_num = fc.out_features
        print('check output_num : {}'.format(self.output_num))
        self.loss_function = nn.CrossEntropyLoss()
        # self.kd_loss_function = nn.KLDivLoss()
        self.finetune_epoch = self.config.finetune_epoch
        self.inference_latency = dict()
        self.retrain_time = dict()
        self.network_latency = dict()
        self.writer = SummaryWriter('logs/IL/')
        
        if self.name == 'googlenet':
            self.input_transform = self.model._transform_input

         
    '''
    Measure the whold latencies on all split point settings.
    1. forward propagation time
    2. retrain time
    3. network latency
    4. reset networks
    '''
    def measure_latency(self, dataloader):
        # tt = 0.
        flag = False
        for split_point in range(totallayer[self.name]):
            head_model = HeadModel(self.name, split_point, self.model).to(self.device)
            tail_model = TailModel(self.name, totallayer[self.name] - fclayer[self.name] + 1 - split_point, self.model).to(self.device)
            print(toBlue(head_model, tail_model))
            tail_optimizer = torch.optim.SGD(tail_model.parameters(), lr=INIT_LR, momentum=0.9)
            # tmp_time = time.perf_counter()
            intermediate_tensors = []
            intermediate_tensor_size = 1
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader):
                    images, _, _ = data
                    images = Variable(images.float()).to(self.device)
                    intermediate_tensor = head_model.forward(images.to(self.device))
                    intermediate_tensors.append(intermediate_tensor)
                    for s in intermediate_tensor.size():
                            intermediate_tensor_size *= s
            for e in range(1):
                e_time = time.perf_counter()
                for batch_idx, data in enumerate(dataloader):
                    t = time.perf_counter()
                    _, targets, _ = data
                    targets = Variable(targets).to(self.device)

                    # intermediate_tensor = head_model.forward(images.to(self.device))
                    # intermediate_tensor_size = 1
                    # for s in intermediate_tensor.size():
                    #     intermediate_tensor_size *= s
                    outputs = tail_model.forward(intermediate_tensors[batch_idx])
                    if self.name == 'googlenet':
                        outputs = outputs[0]
                    inference_latency = time.perf_counter() - t
                    loss = self.loss_function(outputs, targets)

                    loss.backward(retain_graph = True)
                    tail_optimizer.zero_grad()
                    tail_optimizer.step()
                    # retrain_time = time.perf_counter() - t


                    if batch_idx == 0 and e == 0:
                        print('start measure split point {}'.format(split_point))
                        self.network_latency[split_point] = intermediate_tensor_size
                        self.inference_latency[split_point] = time.perf_counter() - t
                        # self.retrain_time[split_point] = time.perf_counter() - t
                    # elif batch_idx == 10:
                    #     self.network_latency[split_point] /= 10.
                    #     self.inference_latency[split_point] /= 10.
                    #     self.retrain_time[split_point] /= 10.
                    #     break
                    else:
                        self.network_latency[split_point] = intermediate_tensor_size
                        self.inference_latency[split_point] += inference_latency
                        # self.retrain_time[split_point] = self.retrain_time[split_point] + retrain_time
                        # tt+=retrain_time
                        # print(retrain_time, self.retrain_time[split_point], tt)
                        # print(type(t), type(time.perf_counter_ns()))
                
                if not flag and e==0:
                    e=0
                    flag = True

                if e==0:
                    self.retrain_time[split_point] = time.perf_counter() - e_time
                else:
                    self.retrain_time[split_point] += time.perf_counter() - e_time
                # print(self.retrain_time)
                # print('###### {}'.format(time.perf_counter() - tmp_time))
                # print(time()-e_time, self.retrain_time)
            # self.inference_latency[split_point] /= float(e)
            # self.retrain_time[split_point] /= float(e)
            

        # self.model.load_state_dict(torch.load('./ckpt/pretrain/' + self.name + '_' + self.dataset + '.pt', map_location = 'cuda'))
        print(toRed('Network Latency Profiling for each split point : ') + toGreen(self.network_latency))
        print(toRed('Inference Latency Profiling for each split point : ') + toGreen(self.inference_latency))
        print(toRed('Retrain Time Profiling for each split point : ') + toGreen(self.retrain_time))
                
        
    '''
    Split the model based on split_point and define the optimizer again.
    '''
    def set_network(self, split_point):
        self.split_point = split_point
        self.head_model = HeadModel(self.name, split_point, self.model).to(self.device)
        self.tail_model = TailModel(self.name, totallayer[self.name] - fclayer[self.name] + 1 - split_point, self.model).to(self.device)
        # self.head_model = torch.nn.Sequential(*self.head_model.model)
        # self.tail_model = torch.nn.Sequential(*self.tail_model.model)
        self.tail_optimizer = torch.optim.SGD(self.tail_model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=5e-4)
        print(toBlue('{}\n{}').format(self.head_model, self.tail_model))

    '''
    Save the model.
    '''
    def save_network(self, dataset, num_task):
        directory = './ckpt/IL/'
        torch.save(self.head_model.state_dict(), directory + self.name + '_head_' + dataset + '_' + str(num_task) + '.pt')
        torch.save(self.tail_model.state_dict(), directory + self.name + '_tail_' + dataset + '_' + str(num_task) + '.pt')       
        


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


    '''Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
    '''
    def distillation_loss(self, y, teacher_scores, T):
        return F.kl_div(F.log_softmax(y / T), F.softmax(teacher_scores / T))

    def concat_models(self):
        torch.save(self.head_model.state_dict(), './head_model.pt')
        torch.save(self.tail_model.state_dict(), './tail_model.pt')
        self.model.load_state_dict(torch.load('./head_model.pt', map_location=self.device), strict=False)
        self.model.load_state_dict(torch.load('./tail_model.pt', map_location=self.device), strict=False)

    '''
    Add new class to self.tail_model.
    '''
    def add_new_class(self, num_new_class):
        if self.name == 'mobilenetv2':
            # add new class
            self.tail_model = self.tail_model.to('cpu')
            # Old number of input/output channel of the last FC layer in old model
            in_features = self.tail_model.classifier[-1].in_features
            out_features = self.tail_model.classifier[-1].out_features
            # Old weight/bias of the last FC layer
            weight = self.tail_model.classifier[-1].weight.data
            bias = self.tail_model.classifier[-1].bias.data
            # New number of output channel of the last FC layer in new model
            new_out_features = num_new_class+out_features
            # Creat a new FC layer and initial it's weight/bias
            new_fc = nn.Linear(in_features, new_out_features)
            xavier_normal_init(new_fc.weight)
            new_fc.weight.data[:out_features] = weight
            new_fc.bias.data[:out_features] = bias
            # Replace the old FC layer
            self.tail_model.classifier[-1] = new_fc
            self.model.classifier[-1] = new_fc
        elif self.name == 'efficientnet_b0':
            # add new class
            self.tail_model = self.tail_model.to('cpu')
            # Old number of input/output channel of the last FC layer in old model
            in_features = self.tail_model._fc.in_features
            out_features = self.tail_model._fc.out_features
            # Old weight/bias of the last FC layer
            weight = self.tail_model._fc.weight.data
            bias = self.tail_model._fc.bias.data
            # New number of output channel of the last FC layer in new model
            new_out_features = num_new_class+out_features
            # Creat a new FC layer and initial it's weight/bias
            new_fc = nn.Linear(in_features, new_out_features)
            xavier_normal_init(new_fc.weight)
            new_fc.weight.data[:out_features] = weight
            new_fc.bias.data[:out_features] = bias
            # Replace the old FC layer
            self.tail_model._fc = new_fc
            self.model._fc = new_fc
        else:
            # add new class
            self.tail_model = self.tail_model.to('cpu')
            # Old number of input/output channel of the last FC layer in old model
            in_features = self.tail_model.fc.in_features
            out_features = self.tail_model.fc.out_features
            # Old weight/bias of the last FC layer
            weight = self.tail_model.fc.weight.data
            bias = self.tail_model.fc.bias.data
            # New number of output channel of the last FC layer in new model
            new_out_features = num_new_class+out_features
            # Creat a new FC layer and initial it's weight/bias
            new_fc = nn.Linear(in_features, new_out_features)
            # print(new_fc.weight)
            # print(new_fc.weight.data)
            # print(new_fc.weight.data.size())
            # print(weight)
            # print(weight.size())
            # xavier_normal_init(new_fc.weight)
            # nn.init.zeros_(new_fc.weight)
            new_fc.weight.data[:out_features,:] = weight[:,:]
            # new_fc.bias.data[:out_features] = bias

            with torch.no_grad():
                old_weights = torch.cat([w for w in self.tail_model.fc.weight])

                old_norm = torch.mean(old_weights.norm())
                new_norm = torch.mean(new_fc.weight.norm())

                new_fc.weight = nn.Parameter((old_norm / new_norm) * new_fc.weight)




            # Replace the old FC layer
            self.tail_model.fc = new_fc
            self.model.fc = new_fc
            # print(self.model.fc.weight)
        # CUDA
        self.tail_model = self.tail_model.to(self.device)
        self.tail_optimizer = torch.optim.SGD(self.tail_model.parameters(), lr=INIT_LR, momentum=0.9, weight_decay=5e-4)
        self.output_num += num_new_class
        
        
    '''
    Migrate the weight and bias in original fc layer to new fc layer.
    And if new class is detected, then add new class into last layer.
    And then retrain the tail model in IL manner.
    You can use this function when measuring retrain one epoch time by setting the epoch param to 1.
    '''
    def incremental_learning(self, dataloader, test_dataloader, epoch, num_new_class, num_task, is_profile=False):
        # torch.autograd.set_detect_anomaly(True)
        # summary(self.tail_model, (64,3,32,32))
        

        # if is_profile:
        self.old_head_model, self.old_tail_model = copy.deepcopy(self.head_model), copy.deepcopy(self.tail_model)
        self.old_head_model.eval()
        self.old_tail_model.eval()


        

        outputs_old = []

        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                images, targets, _ = data
                targets = Variable(targets).cuda(0)
                outputs = self.old_forward(images.to(self.device))
                # print(outputs.size())
                outputs_old.append(outputs)

                    # _, predicted = torch.max(outputs, 1)
                    # correct += (predicted == targets.to(self.device)).sum().item()
                    # total += targets.size(0)
        # print(output_old[0])
        # print(output_old[0].size())

        # migrate the head_model and tail model
        
        # add new class
        self.add_new_class(num_new_class = num_new_class)
        old_output_num = self.output_num - num_new_class


        # log
        if not is_profile:
            writer = SummaryWriter('logs/IL/' + self.dataset + '/' + self.name + '/task' + str(num_task) + '/')
      
        
        retrain_acc = 0
        # t = time()
        for e in range(epoch):
            # if e > 0 :

            correct, total = 0, 0
            correct_label = [0 for i in range(self.output_num)]
            total_label = [0 for i in range(self.output_num)]

            # train
            self.head_model.eval()
            self.tail_model.train()
            # self.tail_optimizer = step_lr(e, 0.001, 50, 0.1, self.tail_optimizer)
            
            for batch_idx, data in enumerate(dataloader):
                # if batch_idx == 5:
                #     input()
                self.tail_optimizer.zero_grad()


                # with torch.no_grad():
                #     images, targets, _ = data
                #     targets = Variable(targets).cuda(0)
                #     outputs = self.old_forward(images.to(self.device))
                #     # print(outputs.size())
                #     outputs_old.append(outputs)



                images, targets, _ = data
                targets = Variable(targets).cuda(0)
                one_hot_targets = F.one_hot(targets, num_classes=self.output_num).float()
                # print(one_hot_targets)
                # outputs = self.forward(images.to(self.device), is_train=True)
                outputs = self.tail_model(images.to(self.device))
                # print('############################################################')
                # print(images)
                # print(outputs)

                # print(outputs)
                # soft_target = self.old_forward(images.to(self.device))
                # import pdb; pdb.set_trace()
                # proba = torch.nan_to_num(F.softmax(outputs, dim=1))
                # old_proba = torch.nan_to_num(F.softmax(outputs_old[batch_idx], dim=1))
                proba = F.softmax(outputs, dim=1)
                old_proba = F.softmax(outputs_old[batch_idx], dim=1)
                # print(proba)
                # print(old_proba)

                # print(old_proba.size())

                # print(batch_idx, proba[..., -num_new_class:].size(), one_hot_targets[..., -num_new_class].size())
                
                CE_loss = F.binary_cross_entropy(proba[..., -num_new_class:], one_hot_targets[..., -num_new_class:])
                # print(proba[..., -num_new_class:].size())
                # CE_loss = self.loss_function(outputs[..., -old_output_num:], targets)

                modified_proba = torch.pow(proba[..., :-num_new_class], 2)
                modified_old_proba = torch.pow(old_proba, 2)
                

                # print(modified_proba)
                # print(modified_old_proba)
                
                # print(modified_proba.size())
                # print(len(modified_proba.sum(-1)))
                # print(modified_old_proba.size())

                modified_proba = modified_proba / modified_proba.sum(-1).view(len(modified_proba.sum(-1)), 1)
                # modified_proba = torch.nan_to_num(modified_proba)
                modified_old_proba = modified_old_proba / modified_old_proba.sum(-1).view(len(modified_old_proba.sum(-1)), 1)
                # modified_old_proba = torch.nan_to_num(modified_old_proba)

                # print(modified_proba)
                # print(modified_old_proba)
                # print(modified_proba)

                KD_loss = 5. * F.binary_cross_entropy_with_logits(modified_proba, modified_old_proba)
                # print()

                # print(CE_loss, KD_loss)

                # if num_task > 0:
                # print(outputs.size())
                # partial_output = outputs[..., :old_output_num]
                # KD_loss = F.kl_div(F.log_softmax(partial_output / self.T), F.softmax(outputs_old[batch_idx] / self.T), reduction = 'batchmean')
                # print(toBlue('KD loss:{}\tCE loss:{}'.format(KD_loss, CE_loss)))

                # loss1 = self.loss_function(outputs, targets)
                # outputs_S = F.softmax(outputs[:,:old_out_features]/self.T,dim=1)
                # outputs_T = F.softmax(soft_target[:,:old_out_features]/self.T,dim=1)

                # loss2 = outputs_T.mul(-1*torch.log(outputs_S))
                # loss2 = loss2.sum(1)
                # loss2 = loss2.mean()*self.T*self.T

                # KD_loss = self.kd_loss_function(F.log_softmax(outputs/self.T, dim=1),
                #             F.softmax(soft_target/self.T, dim=1)) * (self.alpha * self.T * self.T) + \
                #             F.cross_entropy(outputs, targets) * (1. - self.alpha)
                
                # loss = KD_loss*self.alpha + CE_loss*(1-self.alpha)
                # loss = KD_loss

                # if e <= self.finetune_epoch:
                #     loss = CE_loss    
                #     loss.backward(retain_graph=True)
                #     for module in self.tail_model.modules():
                #         if isinstance(module, nn.Conv2d):
                #             module.weight.grad.data.fill_(0)
                # else:
                # if num_task > 0:
                loss = 2.5*KD_loss + CE_loss
                # print(loss)
                # else:
                # loss = KD_loss + CE_loss
                # loss = KD_loss

                # loss.backward(retain_graph=True)
                # print(loss)
                loss.backward()

                self.tail_optimizer.step()
        
                _, predicted = torch.max(outputs, 1)
                # print(predicted, targets)
                correct += (predicted == targets.to(self.device)).sum().item()
                total += targets.size(0)
                for i in range(len(targets)):
                    total_label[targets[i]] += 1
                    if predicted[i] == targets[i]:
                        correct_label[targets[i]] += 1
            
            



            retrain_acc = 100.*correct/total
                # progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #     % (loss.item()/(batch_idx+1), retrain_acc, correct_one_epoch, total_one_epoch))
            if e % 2 == 0:
                with torch.no_grad():
                    in_features = self.tail_model.fc.in_features
                    out_features = self.tail_model.fc.out_features
                    new_fc = nn.Linear(in_features, out_features).to(self.device)
                    new_fc.weight[:out_features,:] = self.tail_model.fc.weight[:,:]
                    old_weights = torch.cat([w for w in self.tail_model.fc.weight])

                    old_norm = torch.mean(old_weights.norm())
                    new_norm = torch.mean(new_fc.weight.norm())

                    new_fc.weight = nn.Parameter((old_norm / new_norm) * new_fc.weight)
                    self.tail_model.fc = new_fc
                print(CE_loss, KD_loss)
                if is_profile:
                    print(toGreen('(Profile) Model: {}\tSplit Point: {}\tRetrain Accuracy: {}'.format(self.name, self.split_point, retrain_acc)))
                else:
                    # if e > 0:
                    print(toGreen('(IL) Epoch: {}\tModel: {}\tSplit Point: {}\tRetrain Accuracy: {}'.format(e, self.name, self.split_point, retrain_acc)))
                    print(toYellow('######### Test Start Task {}\tEpoch {} #########'.format(num_task, e)))
                    self.test(dataloader=test_dataloader, num_task=num_task, epoch = e)

            if not is_profile:
                writer.add_scalar('acc/train', retrain_acc, e)
                if e == epoch -1 and num_new_class == 0:
                    for i in range(self.output_num):
                        print(toGreen('label: {}\taccuracy: {}'.format(i, 100.*correct_label[i]/total_label[i])))

        if not is_profile:
            writer.close()

        # IL_time = time() - t
        self.head_model.eval()    
        self.tail_model.eval()
        self.concat_models()
        return retrain_acc#, IL_time
    
    '''
    test the model.
    '''
    def test(self, dataloader, num_task, epoch):
        writer = SummaryWriter('logs/IL/' + self.dataset + '/' + self.name + '/task' + str(num_task) + '/')
        correct, total = 0, 0
        correct_label = [0 for i in range(self.output_num)]
        total_label = [0 for i in range(self.output_num)]
        print(self.output_num)

        # eval mode
        self.head_model.eval()
        self.tail_model.eval()
       
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                images, targets, _ = data
                outputs = self.forward(images.to(self.device))
        
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets.to(self.device)).sum().item()
                total += targets.size(0)
                        
            for i in range(len(targets)):
                total_label[targets[i]] += 1
                if predicted[i] == targets[i]:
                    correct_label[targets[i]] += 1

        eval_acc = 100.*correct/total
        print(toGreen('Model: {} Split Point: {} Test Accuracy: {}'.
                        format(self.name, self.split_point, eval_acc)))
        
        for i in range(self.output_num):
            writer.add_scalar('acc/test/task{}/label{}'.format(num_task, i), 100.*correct_label[i]/total_label[i], epoch)
            print(toGreen('label: {}\taccuracy: {}/{} = {}'.format(i, correct_label[i], total_label[i], 100.*correct_label[i]/total_label[i])))

        writer.add_scalar('acc/test/task{}'.format(num_task), 100.*correct/total, epoch)
        writer.close()

        return eval_acc