from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from utils.solver import build_optimizer, build_lr_scheduler
from utils.checkpointer import DetectionCheckpointer, PeriodicCheckpointer
from utils.logger import setup_logger
from utils.event import EventStorage, CommonMetricPrinter, TensorboardXWriter
from icarl import Icarl
import torch.nn.functional as F
import utils_func
import time
import torch.nn as nn

np.random.seed(100)



class dynamicDataset(Dataset):
    def __init__(self):
        self.len = 0
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pass


parser = argparse.ArgumentParser(
    description='Context-Transformer')

# Model and Dataset
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset.')
parser.add_argument('--split', type=int, default=1,
                    help='VOC base/novel split, for VOC only.')

# Training Parameters
parser.add_argument('--setting', default='transfer',
                    help='Training setting: transfer or incre.')
parser.add_argument('-p', '--phase', type=int, default=0,
                    help='Training phase. 1: source pretraining, 2: target fintuning.')
parser.add_argument('-m', '--method', default='ours',
                    help='ft(baseline) or ours, for phase 2 only.')
parser.add_argument('--shot', type=int, default=5,
                    help="Number of shot, for phase 2 only.")
parser.add_argument('--init-iter', type=int, default=50,
                    help="Number of iterations for OBJ(Target) initialization")
parser.add_argument('-max', '--max-iter', type=int, default=180000,
                    help='Number of training iterations.')
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='Batch size for training')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.004,
                    help='Initial learning rate')
parser.add_argument('--steps', type=int, nargs='+', default=[120000, 150000],
                    help='Learning rate decrease steps.')
parser.add_argument('--warmup-iter', type=int, default=5000,
                    help='Number of warmup iterations')
parser.add_argument('--ngpu', type=int, default=4, help='gpus')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Use cuda to train model')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum for SGD')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='Gamma update for SGD')
parser.add_argument('--load-file', default=None,
                    help='Model checkpoint for loading.')
parser.add_argument('--resume', action='store_true',
                    help='Whether resume from the last checkpoint.'
                         'If True, no need to specify --load-file.')
parser.add_argument('-is', '--instance-shot', action='store_true',
                    help='If True, instance shot will be applied for transfer setting.')

# TODO
# Mixup
parser.add_argument('--mixup', action='store_true',
                    help='Whether to enable mixup.')
parser.add_argument('--no-mixup-iter', type=int, default=800,
                    help='Disable mixup for the last few iterations.')

# Output
parser.add_argument('--save-folder', default='./weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--checkpoint-period', type=int, default=10000,
                    help='Checkpoint period.')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

logger = setup_logger(args.save_folder)

num_classes = 10
img_dim = (300, 512)[args.size == '512']
rgb_means = (104, 117, 123)
p = 0.6
overlap_threshold = 0.5
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
             'truck': 9}

def data_loader_update(dataset_source, testset_source, iteration):
    # dataset = utils_func.DatasetMaker(
    #     [utils_func.get_class_i(testset_source.data, testset_source.targets, i)[:500] for i in range(iteration )] + [
    #         utils_func.get_class_i(dataset_source.data, dataset_source.targets, iteration)],
    #     torchvision.transforms.ToTensor()
    # )

    dataset = utils_func.DatasetMaker(
        [utils_func.get_class_i(testset_source.data, testset_source.targets, i) for i in range(iteration)] + [
            utils_func.get_class_i(dataset_source.data, dataset_source.targets, iteration)],
        torchvision.transforms.ToTensor()
    )

    testset = utils_func.DatasetMaker(
        [utils_func.get_class_i(testset_source.data, testset_source.targets, i) for i in range(iteration + 1)],
        torchvision.transforms.ToTensor()
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        # args.batch_size,
        128,
        num_workers=args.num_workers,
        shuffle=True
    )
    return data_loader, test_loader


def train(icarl, resume=False):
    model = icarl.model
    if args.cuda and torch.cuda.is_available():
        model.device = 'cuda'
        model.cuda()
        cudnn.benchmark = True
        if args.ngpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    else:
        model.device = 'cpu'

    model.train()
    # print(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)
    checkpointer = DetectionCheckpointer(
        model, args, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(args.basenet if args.phase == 1 else args.load_file,
                                        resume=resume).get("iteration", -1) + 1
    )
    max_iter = args.max_iter
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, args.checkpoint_period, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            TensorboardXWriter(args.save_folder),
        ]
    )

    dataset_source = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    human_dataset = utils_func.DatasetMaker(
        [utils_func.get_class_i(dataset_source.data, dataset_source.targets, i) for i in [46]],
        torchvision.transforms.ToTensor(),
        [46]
    )
    dataset = dataset_source
    testset_source = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # testset = utils_func.DatasetMaker(
    #     [utils_func.get_class_i(testset_source.data, testset_source.targets, i) for i in range(2)],
    #     torchvision.transforms.ToTensor()
    # )
    testset = testset_source

    human_testset = utils_func.DatasetMaker(
        [utils_func.get_class_i(testset.data, testset.targets, i) for i in [46]],
        torchvision.transforms.ToTensor(),
        [46]
    )

    human_test_loader = torch.utils.data.DataLoader(
        human_testset,
        # args.batch_size,
        64,
        num_workers=4,
        # shuffle=True
    )

    human_data_loader = torch.utils.data.DataLoader(
        human_dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        # args.batch_size,
        64,
        num_workers=args.num_workers,
        shuffle=True
    )

    assert model.training, 'Model.train() must be True during training.'
    logger.info("Starting training from iteration {}".format(start_iter))

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.gamma, last_epoch=epoch - 1)
    minor_first = True
    import random
    f = open("/app/ekya/mylog_all_layer.txt", 'w')
    with EventStorage(start_iter) as storage:
        for iteration in range(start_iter, max_iter):
            # print(max_iter)
            iteration = iteration + 1

            storage.step()
            # print(len(data_loader))
            # if iteration > 1 and iteration <= 9:
            #     data_loader, test_loader = data_loader_update(dataset_source, testset_source, iteration)

            # if iteration == 9:
            #     data_loader = torch.utils.data.DataLoader(
            #         dataset_source,
            #         args.batch_size,
            #         num_workers=args.num_workers,
            #         shuffle=True
            #     )
            #     test_loader = torch.utils.data.DataLoader(
            #         testset_source,
            #         # args.batch_size,
            #         128,
            #         num_workers=args.num_workers,
            #         shuffle=True
            #     )
            train_data_loader = data_loader
            if minor_first:
                train_data_loader = human_data_loader
            # train_data_loader = human_data_loader
            traintime = 0
            infertime = 0

            for batch_idx, (data, targets) in enumerate(train_data_loader):
                # data = F.interpolate(data, size=(224, 224))

                if (not minor_first) and random.random()<0.3:
                    continue

                start_t = time.time()

                storage.put_image('image', vis_tensorboard(data))
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                # print(output, targets)
                # exit()
                losses = criterion(output, targets.to('cuda'))
                # print(losses)
                # loss_dict = criterion(output, priors, targets)
                # losses = sum(loss for loss in loss_dict.values())
                # # assert torch.isfinite(losses).all(), loss_dict
                storage.put_scalars(total_loss=losses)
                #
                optimizer.zero_grad()
                infertime += time.time() - start_t

                start_t = time.time()

                losses.backward()
                optimizer.step()
                scheduler.step()
                traintime += time.time() - start_t
            #
            # for batch_idx, (data, targets) in enumerate(human_data_loader):
            #     # data = F.interpolate(data, size=(224, 224))
            #     start_t = time.time()
            #
            #     storage.put_image('image', vis_tensorboard(data))
            #     output = model(data)
            #     criterion = nn.CrossEntropyLoss()
            #     # print(output, targets)
            #     # exit()
            #     losses = criterion(output, targets.to('cuda'))
            #     # print(losses)
            #     # loss_dict = criterion(output, priors, targets)
            #     # losses = sum(loss for loss in loss_dict.values())
            #     # # assert torch.isfinite(losses).all(), loss_dict
            #     storage.put_scalars(total_loss=losses)
            #     #
            #     optimizer.zero_grad()
            #     infertime+=time.time()-start_t
            #
            #     start_t = time.time()
            #
            #     losses.backward()
            #     optimizer.step()
            #     scheduler.step()
            #     traintime += time.time() - start_t

            # print(iteration)

            if True:
                # if iteration - start_iter > 5 and (iteration % 10 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            if True:

                # if iteration - start_iter > 5 and (iteration % 50 == 0 or iteration == max_iter):
                correct = 0
                top5_correct = 0
                total = 0
                top5_humans = 0
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        # images = F.interpolate(images, size=(224, 224))
                        labels = labels.to('cuda')
                        # calculate outputs by running images through the network
                        outputs = model(images.to('cuda'))
                        # the class with the highest energy is what we choose as prediction
                        # print(labels.shape)
                        fake_label = torch.Tensor([46] * labels.shape[0]).to('cuda')
                        top5_humans += utils_func.accuracy(outputs, fake_label, (5,))[0].item()

                        top5_correct += utils_func.accuracy(outputs, labels, (5,))[0].item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        # print(predicted, labels)
                        # break
                        correct += (predicted == labels).sum().item()
                    human_correct = 0
                    human_total = 0
                    human_top5_correct = 0
                    for data in human_test_loader:
                        images, labels = data
                        # images = F.interpolate(images, size=(224, 224))
                        labels = labels.to('cuda')
                        # calculate outputs by running images through the network
                        outputs = model(images.to('cuda'))
                        # the class with the highest energy is what we choose as prediction
                        # print(labels)

                        fake_label = torch.Tensor([46] * labels.shape[0]).to('cuda')

                        human_top5_correct += utils_func.accuracy(outputs, labels, (5,))[0].item()

                        _, predicted = torch.max(outputs.data, 1)
                        human_total += labels.size(0)
                        # print(predicted, labels)
                        # break
                        human_correct += (predicted == labels).sum().item()

                print(
                    'Top1 Accuracy: %d %% Top5 Accuracy: %d %% ' % (100 * correct / total, 100 * top5_correct / total),
                    "{:.3f}".format(traintime), 's', "{:.3f}".format(infertime),
                    'Person on top5: %d %%' % (100 * top5_humans / total))
                print('Top1 Accuracy of person images: %d %% Top5 Accuracy: %d %% ' % (
                    100 * human_correct / human_total, 100 * human_top5_correct / human_total), human_correct,
                      human_total)

                f.write('%d\t%d\t%d\t%d\t%d\n' % (
                (100 * correct / total), (100 * human_correct / human_total), (100 * top5_correct / total),
                (100 * human_top5_correct / human_total), (100 * top5_humans / total)) )
                f.close()
                f = open("/app/ekya/mylog_all_layer.txt", 'a')

                if (correct / total) < (human_correct / human_total):
                    # if (top5_correct / total) < (human_top5_correct / human_total):
                    minor_first = False
                    print('Train all')
                else:
                    minor_first = True
                    print('Train human only')
                model.train()


def vis_tensorboard(images):
    rgb_mean = torch.Tensor(rgb_means).to(images.device)
    image = images[0] + rgb_mean[:, None, None]
    image = image[[2, 1, 0]].byte()
    return image


if __name__ == '__main__':
    icarl = Icarl(size='pretrained', class_num=10)
    model = icarl.model

    logger.info("Model:\n{}".format(model))
    train(icarl, args.resume)
