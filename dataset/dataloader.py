'''
Author: Tung Nguyen (tungnt@kaist.ac.kr)
'''

import os
import sys

from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

try:
    from cifar_pretrain import LIGETIPretrainCIFAR10,\
        LIGETIPretrainCIFAR100
    from imagenet_pretrain import LIGETIPretrainImageNet100
except ModuleNotFoundError as e:
    raise e


class PretrainDataset(Dataset):
    """PretrainDataset prepare the dataset for training

    Parameters
    ----------
    dataset_name : str
        The name of the dataset from which the pretraining data
        will be taken.
        (cifar10, cifar100, imagenet100)
    data_dir_path : str
        The path to the dataset directory. The data should be organized
        into two sub-dirs `train` and `test`, each of which is then
        organized into sub-dirs of classes.
    num_classes_for_pretrain : int
        number of classes that will be used for pretraining
    num_of_imgs_from_chosen_classes : list of tuple
        number of images from each chosen class. Each tuple contains
        (number of imgs, number of classes from each of which this many
        images would be taken)
    train : bool
        Are we training or testing?
    choosing_class_seed : int, optional
        The seed to choose random classes for pretraining
    train_data_shuffle_seed : int, optional
        The seed to choose random data in each class for pretrianing. A
        pair of train seed and test seed should be fixed.
    test_data_shuffle_seed : int, optional
        The seed to choose random data in each class for pretrianing's
        testing. A pair of train seed and test seed should be fixed.
    """
    def __init__(
        self,
        dataset_name: str,
        data_dir_path: str,
        num_classes_for_pretrain: int,
        num_imgs_from_chosen_classes: list,
        train: bool,
        choosing_class_seed: int = 2022,
        train_data_shuffle_seed: int = 222,
        test_data_shuffle_seed: int = 223,
        transform=None,
        target_transform=None
    ) -> None:
        super(PretrainDataset, self).__init__()

        if dataset_name == 'cifar10':
            dataset_preparer = LIGETIPretrainCIFAR10
        elif dataset_name == 'cifar100':
            dataset_preparer = LIGETIPretrainCIFAR100
        elif dataset_name == 'imagenet100':
            dataset_preparer = LIGETIPretrainImageNet100

        self.pretrain_dataset = dataset_preparer(
            data_dir_path,
            num_classes_for_pretrain,
            num_imgs_from_chosen_classes,
            train,
            choosing_class_seed,
            train_data_shuffle_seed,
            test_data_shuffle_seed
        )

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.pretrain_dataset.pretrain_data)

    def __getitem__(self, index):
        img, img_class, img_class_name = self.pretrain_dataset(index)
        if self.transform is not None:
            # print(img.shape)
            img = self.transform(img)
        if self.target_transform is not None:
            img_class = self.target_transform(img_class)
        return img, img_class, img_class_name


if __name__ == '__main__':
    pretrain_dataset = PretrainDataset(
        dataset_name='cifar100',
        data_dir_path='/data/cifar100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (100, 10), (250, 20), (500, 10)
        ],
        train=True,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222
    )
    print(pretrain_dataset[1000])
