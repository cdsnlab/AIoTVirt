'''
Author: Tung Nguyen (tungnt@kaist.ac.kr)
'''

import os
import sys
from PIL import Image

from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

try:
    from cifar_pretrain import LIGETIPretrainCIFAR10,\
        LIGETIPretrainCIFAR100
    from imagenet_pretrain import LIGETIPretrainImageNet100
    from retrain_dataset_preparer import RetrainingDatasetPreparer
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


class RetrainDataset(Dataset):
    def __init__(
        self,
        task_num: int,
        dataset_name: str,
        data_dir_path: str,
        num_total_classes: int,
        num_pretrain_classes: int,
        num_test_images_each_class: int,
        num_total_images_each_task: int,
        task_specifications: list,
        choosing_class_seed: int = 2022,
        retrain_data_shuffle_seed: int = 2,
        pretrain_test_data_shuffle_seed: int = 222,
        transforms=None,
        target_transforms=None,
    ):
        super(RetrainDataset, self).__init__()
        self.task_retrain_data = RetrainingDatasetPreparer(
            dataset_name=dataset_name,
            data_dir_path=data_dir_path,
            num_total_classes=num_total_classes,
            num_pretrain_classes=num_pretrain_classes,
            num_test_images_each_class=num_test_images_each_class,
            num_total_images_each_task=num_total_images_each_task,
            task_specifications=task_specifications,
            choosing_class_seed=choosing_class_seed,
            retrain_data_shuffle_seed=retrain_data_shuffle_seed,
            pretrain_test_data_shuffle_seed=pretrain_test_data_shuffle_seed
        ).all_task_retrain_data[task_num]

        self.transforms = transforms
        self.target_transforms = target_transforms
    
    def __len__(self):
        return len(self.task_retrain_data)

    def __getitem__(self, index):
        img, class_idx, class_name = self.task_retrain_data[index]
        if type(img) is str:
            img = Image.load(img)
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            class_idx = self.target_transforms(class_idx)
        return (img, class_idx, class_name)


if __name__ == '__main__':
    dataset = 'cifar10'
    dataset = RetrainDataset(
        task_num=3,
        dataset_name=dataset,
        data_dir_path='/data/{}/test'.format(dataset),
        num_total_classes=10,
        num_pretrain_classes=4,
        num_test_images_each_class=50,
        num_total_images_each_task=1000,
        task_specifications=[
            (2, 0, 605),
            (0, 6, 570),
            (2, 0, 576),
            (0, 8, 504),
            (2, 0, 604),
            (0, 10, 304)
        ]
    )
    print(len(dataset))
    print(dataset[999])
    # pretrain_dataset = PretrainDataset(
    #     dataset_name='cifar100',
    #     data_dir_path='/data/cifar100',
    #     num_classes_for_pretrain=40,
    #     num_imgs_from_chosen_classes=[
    #         (100, 10), (250, 20), (500, 10)
    #     ],
    #     train=True,
    #     choosing_class_seed=2022,
    #     train_data_shuffle_seed=223,
    #     test_data_shuffle_seed=222
    # )
    # cifar10_train_dataset = PretrainDataset(
    #     dataset_name='cifar10',
    #     data_dir_path='/data/cifar10',
    #     num_classes_for_pretrain=4,
    #     num_imgs_from_chosen_classes=[
    #         (500, 1), (1000, 1), (1500, 1), (2000, 1)
    #     ],
    #     train=True,
    #     choosing_class_seed=2022,
    #     train_data_shuffle_seed=223,
    #     test_data_shuffle_seed=222,
    # )
    # imagenet100_train_dataset = PretrainDataset(
    #     dataset_name='imagenet100',
    #     data_dir_path='/data/imagenet100',
    #     num_classes_for_pretrain=40,
    #     num_imgs_from_chosen_classes=[
    #         (100, 10), (250, 20), (500, 10)
    #     ],
    #     train=True,
    #     choosing_class_seed=2022,
    #     train_data_shuffle_seed=223,
    #     test_data_shuffle_seed=222,
    # )
    # print(pretrain_dataset[1000])
