import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import torch.nn.functional as F
import torchvision


class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, label_file=None, transform=None, train = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.label = None
        if label_file:
            if train:
                self.label = pd.read_csv(label_file, nrows=19000)
                self.landmarks_frame = pd.read_csv(csv_file, nrows=19000)
            else:
                self.label = pd.read_csv(label_file, skiprows=range(1, 19000))
                self.landmarks_frame = pd.read_csv(csv_file, skiprows=range(1, 19000))
        else:
                self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)

        #image = np.vstack(image)
        #image = image.transpose((0, 2, 3, 1))
        #sample = (image, self.label.iloc[idx, 0])
        image =torchvision.transforms.ToTensor()(image)

        image = torch.unsqueeze(image,0)

        #image = self.transform(image)
        image = F.interpolate(image, size=(64,64)).reshape((-1,64,64))
        sample = (image, 1)
        if self.label is not None:

            sample = (image, self.label.iloc[idx, 0])
            #sample = (image, self.label.iloc[idx, 0])

        return sample
