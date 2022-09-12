
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
from torchvision.models import resnet18, ResNet18_Weights

INIT_LR = 1e-3
# INIT_LR = 0
print('initial lr {}'.format(INIT_LR))

'''
The information of layer in each model is listed in dictionary.
These are used when splitting the model.
'''

fclayer = {'resnet18': 1, 'mobilenetv2': 1, 'efficientnet_b0': 1, 'shufflenetv2': 1}
totallayer = {'resnet18': 14, 'mobilenetv2': 20, 'efficientnet_b0': 10, 'shufflenetv2': 24}

'''
To make the right size of tensor before fc layer
'''
def pre_fc(x, name, train=False):
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
        if self.name =='mobilenetv2':
            self.fc = copy.deepcopy(self.original_model.classifier[-1])
        elif self.name == 'efficientnet_b0':
            self._fc = copy.deepcopy(self.original_model._fc)
        else:
            self.fc = copy.deepcopy(self.original_model.fc)

        if self.layernum == 1:
            if self.name == 'efficientnet_b0':
                self._fc = copy.deepcopy(self.original_model._fc)
            else:
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
            if self.name == 'efficientnet_b0':
                x = self._fc(x)
            else:
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
        model, fc, _, _, input_transform = model_spec(self.name, self.dataset)
        model.load_state_dict(torch.load('./ckpt/pretrain/' + self.name + '_' + self.dataset + '.pt', map_location = 'cpu'))
        # model, fc, _, _, _ = model_spec(self.name, 'imagenet100')
        # model.load_state_dict(torch.load('./ckpt/pretrain/' + self.name + '_' + 'imagenet100' + '.pt', map_location = 'cpu'))
        # model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # fc = nn.Linear(fc.in_features, 100)
        print('All keys matched')
        self.input_transform = input_transform
        print(self.input_transform)

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
        self.loss_function = nn.CrossEntropyLoss()
        self.kd_loss_function = nn.KLDivLoss()
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
        print(self.model)
        # tt = 0.
        flag = False
        for split_point in range(0, totallayer[self.name]):
            head_model = HeadModel(self.name, split_point, self.model).to(self.device)
            tail_model = TailModel(self.name, totallayer[self.name] - fclayer[self.name] + 1 - split_point, self.model).to(self.device)
            print(toGreen(head_model), toBlue(tail_model))
            tail_optimizer = torch.optim.SGD(tail_model.parameters(), lr=INIT_LR, momentum=0.9)
            # print(tail_model.parameters())
            intermediate_tensors = []
            intermediate_tensor_size = 1
            with torch.no_grad():
                for batch_idx, data in enumerate(dataloader):
                    images, _ = data
                    # print(images)
                    # break
                    # images = Variable(images.float()).to(self.device)
                    # print(self.model(images))
                    intermediate_tensor = head_model.forward(images.float().to(self.device))
                    # intermediate_tensor = head_model.forward(self.input_transform(images.to(self.device)))
                    intermediate_tensors.append(intermediate_tensor)
                    if batch_idx == 0:
                        for s in intermediate_tensor.size():
                            intermediate_tensor_size *= s
            for e in range(1):
                e_time = time.perf_counter()
                for batch_idx, data in enumerate(dataloader):
                    t = time.perf_counter()
                    _, targets = data
                    targets = Variable(targets).to(self.device)
                    # print(intermediate_tensors[batch_idx].size())
                    # self.model(intermediate_tensors[batch_idx])

                    outputs = tail_model.forward(intermediate_tensors[batch_idx])

                    if self.name == 'googlenet':
                        outputs = outputs[0]
                    inference_latency = time.perf_counter() - t
                    loss = self.loss_function(outputs, targets)

                    loss.backward(retain_graph = True)
                    tail_optimizer.zero_grad()
                    tail_optimizer.step()


                    if batch_idx == 0 and e == 0:
                        print('start measure split point {}'.format(split_point))
                        self.network_latency[split_point] = intermediate_tensor_size
                        self.inference_latency[split_point] = time.perf_counter() - t
                    else:
                        self.network_latency[split_point] = intermediate_tensor_size
                        self.inference_latency[split_point] += inference_latency
                
                if not flag and e==0:
                    e=0
                    flag = True

                if e==0:
                    self.retrain_time[split_point] = time.perf_counter() - e_time
                else:
                    self.retrain_time[split_point] += time.perf_counter() - e_time
            

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
        # print(toBlue('{}\n{}').format(self.head_model, self.tail_model))

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
        output = self.tail_model.forward(self.head_model.forward(x))
        return output
    
    '''
    Old model version of forward function.
    '''
    def old_forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.old_head_model.forward(x.float())
            output = self.old_tail_model.forward(output)
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
    Migrate the weight and bias in original fc layer to new fc layer.
    And then retrain the tail model in IL manner.
    You can use this function when measuring retrain_one_epoch_time by setting the epoch param to 1.
    '''
    def incremental_learning(self, dataloader, test_dataloader, epoch, allocated_time, num_task, is_profile=False):
        # if is_profile:
        self.old_head_model, self.old_tail_model = copy.deepcopy(self.head_model), copy.deepcopy(self.tail_model)
        # self.old_head_model.eval()
        # self.old_tail_model.eval()


        # if is_profile:
        #     self.tail_optimizer = torch.optim.SGD(self.tail_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

        # log
        if not is_profile:
            writer = SummaryWriter('logs/IL/' + self.dataset + '/' + self.name + '/task' + str(num_task) + '/')
    
        self.test(dataloader=test_dataloader, num_task=num_task, epoch = 0)

        t = time.perf_counter()
        end_flag = False
        '''
        Make output of the old version model before retraining.
        This can be changed to the grpc-based implementation.
        '''
        outputs_old = []
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                images, targets = data
                targets = Variable(targets).cuda(self.device)
                outputs = self.old_forward(images.to(self.device))
                outputs_old.append(outputs)
        
        retrain_acc = 0
        for e in range(epoch):

            correct, total = 0, 0
            correct_label = [0 for i in range(self.output_num)]
            total_label = [0 for i in range(self.output_num)]

            # train
            # self.head_model.eval()
            self.head_model.train()
            self.tail_model.train()
            
            for batch_idx, data in enumerate(dataloader):
                self.tail_optimizer.zero_grad()

                images, targets = data
                # print(images)
                images, targets = Variable(images.float()).cuda(self.device), Variable(targets).cuda(self.device)

                outputs = self.forward(images)

                KD_loss = self.kd_loss_function(F.log_softmax(outputs/self.T, dim=1), F.softmax(outputs_old[batch_idx]/self.T, dim=1)) * (self.T * self.T)
                CE_loss = F.cross_entropy(outputs, targets) * (1. - self.alpha)
                # print(KD_loss, CE_loss, KD_loss + CE_loss)
                loss = KD_loss + CE_loss
                # loss = CE_loss

                loss.backward()

                self.tail_optimizer.step()
        
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                for i in range(len(targets)):
                    total_label[targets[i]] += 1
                    if predicted[i] == targets[i]:
                        correct_label[targets[i]] += 1

                if time.perf_counter() - t > allocated_time:
                    end_flag = True
                    break


            retrain_acc = 100.*correct/total
            
            
            if e % 1 == 0:
                # with torch.no_grad():
                #     in_features = self.tail_model.fc.in_features
                #     out_features = self.tail_model.fc.out_features
                #     new_fc = nn.Linear(in_features, out_features).to(self.device)
                #     new_fc.weight[:out_features,:] = self.tail_model.fc.weight[:,:]
                #     old_weights = torch.cat([w for w in self.tail_model.fc.weight])

                #     old_norm = torch.mean(old_weights.norm())
                #     new_norm = torch.mean(new_fc.weight.norm())

                #     new_fc.weight = nn.Parameter((old_norm / new_norm) * new_fc.weight)
                #     self.tail_model.fc = new_fc
                    
                print(toRed('Remained time : {}'.format(allocated_time - (time.perf_counter() - t))))
                if is_profile:
                    print(toGreen('(Profile) Model: {}\tSplit Point: {}\tRetrain Accuracy: {}'.format(self.name, self.split_point, retrain_acc)))
                else:
                    # if e > 0:
                    print(toGreen('(IL) Epoch: {}\tModel: {}\tSplit Point: {}\tRetrain Accuracy: {}'.format(e, self.name, self.split_point, retrain_acc)))
                    print(toYellow('######### Test Start Task {}\tEpoch {} #########'.format(num_task, e)))
                    self.test(dataloader=test_dataloader, num_task=num_task, epoch = e)

            if not is_profile:
                writer.add_scalar('acc/train', retrain_acc, e)

            if end_flag:
                break

        if not is_profile:
            writer.close()
            self.concat_models()
        else:
            self.head_model = self.old_head_model
            self.tail_model = self.old_tail_model
            

        # IL_time = time() - t
        self.head_model.eval()    
        self.tail_model.eval()
        return retrain_acc#, IL_time
    
    '''
    test the model.
    '''
    def test(self, dataloader, num_task, epoch):
        writer = SummaryWriter('logs/IL/' + self.dataset + '/' + self.name + '/task' + str(num_task) + '/')
        correct, total = 0, 0
        correct_label = [0 for i in range(self.output_num)]
        total_label = [0 for i in range(self.output_num)]
        # print(self.output_num)

        # eval mode
        self.head_model.eval()
        self.tail_model.eval()
    
        for batch_idx, data in enumerate(dataloader):
            with torch.no_grad():
                images, targets = data
                # print(images)
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

        writer.add_scalar('acc/test/IL/task{}'.format(num_task), 100.*correct/total, epoch)
        writer.close()

        return eval_acc