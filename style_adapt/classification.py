import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
from config import get_opts
from tools.time import Timer

from .network.wide_resnet import wide_resnet_28_10

from robustbench.data import load_cifar10c, load_cifar100c, load_imagenetc

CORRUPTIONS = ["gaussian_noise"]
# CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]


def main():
    opts = get_opts('configs/classification.yaml')

    print(f"Classification on {opts.data}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    dataset_name = opts.data.lower()
    model = wide_resnet_28_10()

    model = model.to(device)
    model.eval()

    severities = [5]
    num_samples = 10000
    avg_err = 0
    cur_iter = 0
    
    timer = Timer()
    timer.tick()
    for i_s, severity in enumerate(severities):
        with tqdm(total=num_samples * len(CORRUPTIONS)) as pbar:
            for ic, corruption in enumerate(CORRUPTIONS):
                x_test, y_test = load_cifar10c(num_samples, severity, opts.datasets[opts.data].path, shuffle=False, corruptions=[corruption])
                dataset = TensorDataset(x_test.cuda(), y_test.cuda())
                dataloader = DataLoader(dataset, batch_size=opts.batch_size)
                tqdm_dataloader = tqdm(dataloader)
                print(f'[{corruption}{severity}] {x_test.shape=}, {y_test.shape=}\n')


                accuracy = 0

                for id, (x, y) in enumerate(tqdm_dataloader):
                    outputs = model(x, warning=False)

                    acc = (outputs.max(dim=1)[1] == y).float().sum()
                    accuracy += acc
                    cur_iter += opts.batch_size
                    
                    tqdm_dataloader.set_postfix(dict(
                        acc = f'{int(accuracy)}/{(id+1) * opts.batch_size} ({accuracy*100/(id+1)/opts.batch_size:.1f}%)'
                    ))
                
                accuracy /= x_test.shape[0]
                err = 1. - accuracy
                print(f'[{corruption}{severity}] acc={accuracy:.3f}, err={err:.3f}')

                avg_err += err
                timer.tick()
        
            print(f'**{severity}: avg error rate = {avg_err / len(CORRUPTIONS):.4f}')
            avg_err = 0
    print("** Time: ", timer.performances(verbose=True))
    print("** Time(>10): ", timer.performances(offset=10, verbose=True))
    print("** Memory: ", torch.cuda.max_memory_allocated(device=device))




if __name__ == '__main__':
    with torch.no_grad():
        main()