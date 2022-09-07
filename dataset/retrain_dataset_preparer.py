'''
Author: Tung Nguyen (tungnt@kaist.ac.kr)
'''

import os
import sys
import random
from pickle import load
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)
sys.path.append(BASE_DIR)
try:
    from dataset.cifar_pretrain import LIGETIPretrainCIFAR10, \
        LIGETIPretrainCIFAR100
    from dataset.imagenet_pretrain import LIGETIPretrainImageNet100
except ModuleNotFoundError:
    raise


class RetrainingDatasetPreparer(Dataset):
    """RetrainingDatasetPreparer Prepares the retrain dataset given
    the full dataset and the specifications that have been used for
    pretraining.

    The full dataset is organized as a dictionary of list, which allows
    access to each individual class. Data of each class is a list in the
    following format.

    |           |                                                  |      |
       pretrain                      retrain                         test

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    data_dir_path : str
        The path to the dataset folder
    num_classes_for_pretrain : int
        Number of classes that have been used for pretraining
    num_imgs_from_chosen_pretrain_classes : list
        The specifications used to construct the pretraining dataset
        E.g., (500, 3), (1000, 3), (1500, 2), (2000, 2)
        This produces 3 classes each with 500 images, 3 classes each with 1000
        images, and so on.
    num_imgs_from_chosen_test_classes : list
        The specifications used to construct the test dataset
        E.g., (50, 10)
    choosing_class_seed : int, optional
        Seed to randomly pick the classes during pretraining, by default 2022
    pretrain_train_data_shuffle_seed : int, optional
        Seed to randomly shuffle the data during pretraining_, by default 222
    pretrain_test_data_shuffle_seed : int, optional
        Seed to randomly shuffle the test data, by default 223
    task_num : int, optional
        The retraining task number, by default 0
    task_specifications : list, optional
        Specifications to construct each task's retraining data
        E.g., task_specifications=[
            (10, 100000, 810)
        ].
        Here, we have 1 task that has a total of 100000 images from 10 classes.
        A seed of 810 will be used to decide the specific number of images in
        each class.
    retrain_data_shuffle_seed : int, optional
        Seed to shuffle the retraining dataset, by default 2
    """
    pretrain_data_preparer_dict = {
        'cifar10': LIGETIPretrainCIFAR10,
        'cifar100': LIGETIPretrainCIFAR100,
        'imagenet100': LIGETIPretrainImageNet100
    }

    def __init__(
        self,
        dataset_name: str,
        data_dir_path: str,
        # The next few parameters are used to construct the pretrain dataset
        # So we can get the remaining data for retraining.
        num_classes_for_pretrain: int,
        num_imgs_from_chosen_pretrain_classes: list,
        num_imgs_from_chosen_test_classes: list,
        choosing_class_seed: int = 2022,
        pretrain_train_data_shuffle_seed: int = 222,
        pretrain_test_data_shuffle_seed: int = 223,
        # The remaining parameters are used to construct the retraining
        # datasets each of which corresponds to a retraining task
        task_num: int = 0,
        task_specifications: list = None,
        retrain_data_shuffle_seed: int = 2,
        transforms=None,
        target_transforms=None
    ):
        self.dataset_name = dataset_name
        self.data_dir_path = data_dir_path
        self.num_classes_for_pretrain = num_classes_for_pretrain
        self.num_imgs_from_chosen_pretrain_classes = \
            num_imgs_from_chosen_pretrain_classes
        self.num_imgs_from_chosen_test_classes = \
            num_imgs_from_chosen_test_classes
        self.choosing_class_seed = choosing_class_seed
        self.pretrain_train_data_shuffle_seed = \
            pretrain_train_data_shuffle_seed
        self.pretrain_test_data_shuffle_seed = pretrain_test_data_shuffle_seed
        self.retrain_data_shuffle_seed = retrain_data_shuffle_seed
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.get_remaining_avail_data_for_retrain()

        specs = task_specifications[task_num]
        num_classes, total_num_images, seed = specs
        num_images_per_classes = \
            self.match_class_data_to_total_data(
                num_classes,
                total_num_images,
                seed
            )
        self.task_retrain_data = self.extract_data_for_retrain_task(
            num_images_per_classes
        )

    def get_remaining_avail_data_for_retrain(self):
        """get_remaining_avail_data_for_retrain Gets all the available data for
        retraining after cutting away the portions used for pretrianing and
        test
        """

        pretrain_data_preparer = \
            self.pretrain_data_preparer_dict[self.dataset_name]
        pretrain_dataset = pretrain_data_preparer(
            self.data_dir_path,
            self.num_classes_for_pretrain,
            self.num_imgs_from_chosen_pretrain_classes,
            True,
            self.choosing_class_seed,
            self.pretrain_train_data_shuffle_seed,
            self.pretrain_test_data_shuffle_seed
        )
        self.full_retrain_avail_dataset = \
            pretrain_dataset.remaining_data
        self.chosen_classes = pretrain_dataset.chosen_classes
        del pretrain_dataset
        for cls in self.chosen_classes:
            self.full_retrain_avail_dataset[cls] = \
                self.full_retrain_avail_dataset[cls][:-50]

    def extract_data_for_retrain_task(
        self,
        num_images_per_classes: list
    ):
        """extract_data_for_retrain_task Extract the data from each class
        and concat them together into one list

        Parameters
        ----------
        num_images_per_classes : list of tuple
            The number of images per individual class

        Returns
        -------
        list
            A list of all data that will be used for retraining
        """
        task_retrain_data = []
        for number, clas in zip(num_images_per_classes, self.chosen_classes):
            class_chosen_data = self.full_retrain_avail_dataset[clas][:number]
            class_chosen_data = [(i, clas) for i in class_chosen_data]
            self.full_retrain_avail_dataset[clas] = \
                self.full_retrain_avail_dataset[clas][number:]
            random.seed(self.retrain_data_shuffle_seed)
            task_retrain_data.extend(class_chosen_data)

        return task_retrain_data

    def match_class_data_to_total_data(
        self,
        num_classes: int,
        total_num_images: int,
        seed: int,
        num_images_each_class_thrs_low: float = 0.6,
        num_images_each_class_thrs_high: float = 1.2,
    ):
        """match_class_data_to_total_data Match the sum of the numbers of
        images from individual classes to a fix number of total images for
        retraining.

        The important constrain is that the number of images from each class
        should be within:
        (avg_num_each_class*num_images_each_class_thrs_low,
        avg_num_each_class*num_images_each_class_thrs_high)
        That is to avoid severe class imbalance.

        Parameters
        ----------
        num_classes : int
            The number of classes that will be incrementally learned during
            this retrianing task.
        num_images_each_class_thrs_low : int
            The ratio between the average number of images and the lower
            threshold
        num_images_each_class_thrs_high : int
            The ratio between the average number of images and the upper
            threshold
        seed : int
            The seed to generate the number of images for each class for this
            task
        total_num_images : int
            The total number of images used for this retraining task
        """
        avg_num_each_class = total_num_images // num_classes
        nums_images_per_classes = []
        random.seed(seed)
        for _ in range(int(num_classes/2)):
            num_image_this_class = random.randint(
                int(avg_num_each_class*num_images_each_class_thrs_low),
                avg_num_each_class
            )
            nums_images_per_classes.append(num_image_this_class)
        for _ in range(int(num_classes/2)+1, num_classes):
            num_image_this_class = random.randint(
                avg_num_each_class,
                int(avg_num_each_class*num_images_each_class_thrs_high),
            )
            nums_images_per_classes.append(num_image_this_class)
        nums_images_per_classes.append(total_num_images -
                                       sum(nums_images_per_classes))

        # random.seed(seed)
        random.shuffle(nums_images_per_classes)
        return nums_images_per_classes

    def __getitem__(self, idx):
        if self.dataset_name == 'imagenet100':
            path, chosen_class = self.task_retrain_data[idx]
            img = Image.open(path)
        elif 'cifar' in self.dataset_name:
            img, chosen_class = self.task_retrain_data[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            chosen_class = self.target_transforms(chosen_class)
        return img, chosen_class

    def __len__(self):
        return len(self.task_retrain_data)


if __name__ == '__main__':
    dataset = 'cifar10'
    temp = RetrainingDatasetPreparer(
        dataset_name=dataset,
        data_dir_path='/data/{}'.format(dataset),
        num_classes_for_pretrain=10,
        num_imgs_from_chosen_pretrain_classes=[
            (500, 3), (1000, 3), (1500, 2), (2000, 2)
        ],
        num_imgs_from_chosen_test_classes=[
            (50, 10)
        ],
        choosing_class_seed=2022,
        pretrain_train_data_shuffle_seed=223,
        pretrain_test_data_shuffle_seed=222,
        task_specifications=[
            (10, 100000, 810)
        ],
        retrain_data_shuffle_seed=2,
    )
    dataloader = DataLoader(
        temp,
        batch_size=32,
        shuffle=True
    )
    for sample in dataloader:
        print(sample)

    # dataset = 'imagenet100'
    # temp = RetrainingDatasetPreparer(
    #     dataset_name=dataset,
    #     data_dir_path='/data/imagenet100',
    #     num_classes_for_pretrain=40,
    #     num_imgs_from_chosen_pretrain_classes=[
    #         (100, 10), (250, 20), (500, 10)
    #     ],
    #     num_imgs_from_chosen_test_classes=[
    #         (50, 10)
    #     ],
    #     choosing_class_seed=2022,
    #     pretrain_train_data_shuffle_seed=223,
    #     pretrain_test_data_shuffle_seed=222,
    #     task_specifications=[
    #         (40, 10000, 810)
    #     ],
    #     retrain_data_shuffle_seed=2,
    # )
    # print(temp[1000])
