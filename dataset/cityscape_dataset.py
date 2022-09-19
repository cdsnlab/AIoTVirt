import os
import sys
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pickle as pkl
from PIL import Image

label_dict = {'person':0, 'car':1, 'bus':2, 'motorcycle':3, 'bicycle':4, 'traffic light':5, 'truck':6, 'stop sign':7, 'train':8}
# PATH = '/home/nahappy15/sdb/cropped_leftImg8bit/'

class CityscapeDataset(Dataset):
    """Cityscape Dataset.

    Parameters
    ----------
    data_dir_path : str
        The root path to the dataset directory.
    city : str
        name of city
    train : bool
        train or not. front part of sequential data
    pretrain : bool
        can access to the whole data
    task : int
        task number
    total_task : int
        total number of tasks
    transform : torchvision.transforms
        return transformed data
    """    
    
    def __init__(
        self,
        data_dir_path: str,
        city: str,
        train: bool,
        pretrain: bool,
        task: int,
        total_task: int,
        transform=None,
        train_data_shuffle_seed: int = 222
    ) -> None:

        self.transform = transform
        self.train_data_shuffle_seed = train_data_shuffle_seed
        
        self.data, self.label = self.read_data(
            data_dir_path=data_dir_path,
            city=city,
            train=train,
            pretrain=pretrain,
            task=task,
            total_task=total_task,
        )
        
        self.chosen_data = []
        for idx in range(len(self.data)):
            self.chosen_data.append((self.data[idx], self.label[idx]))
        del self.data
        del self.label
    
    def read_data(
        self,
        data_dir_path: str,
        city: str,
        train: bool,
        pretrain: bool,
        task: int,
        total_task: int,
    ):
        """read_data _summary_
        """

        full_data = []
        full_label = []
        
        if pretrain:
            city_data_dir_path = os.path.join(data_dir_path, city)
            frames = sorted(os.listdir(city_data_dir_path))
            for frame_idx in range(len(frames)):
                frame_path = os.path.join(city_data_dir_path, str(frame_idx))
                crops = sorted(os.listdir(frame_path))
                del crops[-1]
                labels = None
                with open(os.path.join(frame_path, 'labels.pkl'), 'rb') as f:
                    labels = pkl.load(f)
                for crop_idx in range(len(crops)):
                    full_data.append(Image.open(os.path.join(frame_path, crops[crop_idx])).convert('RGB'))
                    full_label.append(label_dict[labels[crop_idx]])
                        
        else:
            city_data_dir_path = os.path.join(data_dir_path, city)
            frames = sorted(os.listdir(city_data_dir_path))
            task_dist = []
            for i in range(total_task):
                if i == 0:
                    task_dist.append(int(len(frames) * 0.4))
                else:
                    task_dist.append(int(len(frames) * 0.6 / total_task))
            task_dir_num = [0]
            for i in range(total_task):
                task_dir_num.append(task_dir_num[i]+task_dist[i])
            if train:
                for frame_idx in range(task_dir_num[task], int(task_dir_num[task+1] - 0.1*task_dist[task])):
                    frame_path = os.path.join(city_data_dir_path, str(frame_idx))
                    crops = sorted(os.listdir(frame_path))
                    del crops[-1]
                    labels = None
                    with open(os.path.join(frame_path, 'labels.pkl'), 'rb') as f:
                        labels = pkl.load(f)
                    for crop_idx in range(len(crops)):
                        full_data.append(Image.open(os.path.join(frame_path, crops[crop_idx])).convert('RGB'))
                        full_label.append(label_dict[labels[crop_idx]])
                
            else:
                for frame_idx in range(int(task_dir_num[task+1] - 0.1*task_dist[task]), task_dir_num[task+1]):
                    frame_path = os.path.join(city_data_dir_path, str(frame_idx))
                    crops = sorted(os.listdir(frame_path))
                    del crops[-1]
                    labels = None
                    with open(os.path.join(frame_path, 'labels.pkl'), 'rb') as f:
                        labels = pkl.load(f)
                    for crop_idx in range(len(crops)):
                        full_data.append(Image.open(os.path.join(frame_path, crops[crop_idx])).convert('RGB'))
                        full_label.append(label_dict[labels[crop_idx]])
                        
        return full_data, full_label
                
    

    def __len__(self):
        return len(self.chosen_data)

    def __getitem__(self, index):
        img, img_class = self.chosen_data[index]
        if self.transform is not None:
            # print(img.shape)
            img = self.transform(img)
        return img, img_class

if __name__ == '__main__':
    dataset = CityscapeDataset(
        data_dir_path= '/home/nahappy15/sdb/cropped_leftImg8bit/',
        city = 'berlin',
        train = True,
        pretrain = True,
        task = 1,
        total_task = 5,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64,64)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.Normalize(
                [0.48560741861744905, 0.49941626449353244, 0.43237713785804116],
                [0.2321024260764962, 0.22770540015765814, 0.2665100547329813])])
    ) 
    print(len(dataset))
    # print(dataset.chosen_data)
    dataloader = DataLoader(
        dataset,
        64,
        num_workers = 8,
        # shuffle=True
    )
    
    for batch_idx, data in enumerate(dataloader):
        images, targets = data
        print(targets)