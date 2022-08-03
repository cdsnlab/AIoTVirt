import numpy as np
import operator
import collections
import os
import fnmatch
import termcolor
import time
import string
import math
from shutil import rmtree
from pathlib import Path
import sys
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    #for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
    #    sys.stdout.write(' ')

    # Go back to the center of the bar.
    #for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
    #    sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
def get_file_path(path, regex):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for i in np.arange(len(regex)):
            for filename in fnmatch.filter(filenames, regex[i]):
                file_path.append(os.path.join(root, filename))

    return file_path

def remove_file_end_with(path, regex):
    file_paths = get_file_path(path, [regex])

    for i in np.arange(len(file_paths)):
        os.remove(file_paths[i])

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])