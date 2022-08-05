'''
Author: Tung Nguyen (tungnt@kaist.ac.kr)
'''

import os
import sys
from pickle import load
import random
import tracemalloc

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)


class LIGETIPretrainCIFAR10(object):
    """LIGETIPretrainCIFAR10 Prepare pretraining data for LIGETI. The data is
    taken from the CIFAR10 dataset.

    For pretraining only about 40% of classes are used. And from each class,
    Only a small proportion of data (10-40%) will be used for pretraining.

    Parameters
    ----------
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
        The seed to choose random data in each class for pretrianing
    test_data_shuffle_seed : int, optional
        The seed to choose random data in each class for pretrianing's
        testing
    """

    total_num_classes = 10

    def __init__(
        self,
        data_dir_path: str,
        num_classes_for_pretrain: int,
        num_imgs_from_chosen_classes: list,
        train: bool,
        choosing_class_seed: int = 2022,
        train_data_shuffle_seed: int = 222,
        test_data_shuffle_seed: int = 223
    ) -> None:

        if train:
            dset = 'train'
            data_seed = train_data_shuffle_seed
        else:
            dset = 'test'
            data_seed = test_data_shuffle_seed
        data_dir_path = os.path.join(data_dir_path, dset)

        random.seed(choosing_class_seed)
        classes_chosen_for_pretrain = random.sample(
            range(self.total_num_classes), num_classes_for_pretrain)
        classes_chosen_for_pretrain = sorted(classes_chosen_for_pretrain)
        print(classes_chosen_for_pretrain)

        data = {}
        for class_data_file_name in os.listdir(data_dir_path):
            class_idx = int(class_data_file_name.split('.')[0])
            class_data_file_path = os.path.join(
                data_dir_path,
                class_data_file_name
            )
            with open(class_data_file_path, 'rb') as f:
                class_data = load(f)
                f.close()
                random.seed(data_seed)
                random.shuffle(class_data)
                data[class_idx] = class_data
        chosen_idx = 0
        self.pretrain_data = []
        for num_chosen_imgs, num_classes_for_num_imgs in\
                num_imgs_from_chosen_classes:
            for clas in range(num_classes_for_num_imgs):
                chosen_class = classes_chosen_for_pretrain[chosen_idx]
                chosen_data = data[chosen_class][:num_chosen_imgs]
                chosen_data = [
                    (x, chosen_idx, chosen_class) for x in chosen_data
                ]
                self.pretrain_data.extend(chosen_data)
                chosen_idx += 1
        del chosen_data
        del class_data
        del data

    def __call__(self, idx):
        """__call__ Get an item from the training data list given its
        index. This function should be called inside the __getitem__
        function provided for torch Dataset.

        Parameters
        ----------
        idx : int
            index of the item
        Returns
        -------
        (np.darray, int, str)
            an image of the dataset in un-preprocessed format, shaped
            (32, 32, 3) and its class and the class's name
        """
        # print(self.pretrain_data[idx][0].shape)
        return self.pretrain_data[idx]


class LIGETIPretrainCIFAR100(LIGETIPretrainCIFAR10):
    """LIGETIPretrainCIFAR10 Prepare pretraining data for LIGETI. The data is
    taken from the CIFAR10 dataset.

    For pretraining only about 40% of classes are used. And from each class,
    Only a small proportion of data (10-40%) will be used for pretraining.

    Parameters
        ----------
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
            The seed to choose random data in each class for pretrianing
        test_data_shuffle_seed : int, optional
            The seed to choose random data in each class for pretrianing's
            testing
    """
    total_num_classes = 100


if __name__ == '__main__':
    # LIGETIPretrainCIFAR10(
    #     data_dir_path='/home/hihi/LIGETI/dataloader/cifar10/train',
    #     num_classes_for_pretrain=4,
    #     num_imgs_from_chosen_classes=[
    #         (500, 1), (1000, 1), (1500, 1), (2000, 1)
    #     ],
    #     train=True,
    #     seeds=(222, 2022)
    # )
    tracemalloc.start()
    temp = LIGETIPretrainCIFAR100(
        data_dir_path='/data/cifar100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (20, 40)
            #(50, 10), (150, 20), (200, 10)
        ],
        train=True,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222
    )
    temp(10)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
