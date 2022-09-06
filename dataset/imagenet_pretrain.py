'''
Author: Tung Nguyen (tungnt@kaist.ac.kr)
'''

import os
import sys
import random
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)
sys.path.append(BASE_DIR)

try:
    from dataset.cifar_pretrain import LIGETIPretrainCIFAR10
except ModuleNotFoundError as e:
    raise e


class LIGETIPretrainImageNet100(LIGETIPretrainCIFAR10):
    """LIGETIPretrainImageNet100 Prepare pretraining data for LIGETI. The data
    is taken from the ImageNet100 dataset.

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
        The seed to choose random data in each class for pretrianing. A
        pair of train seed and test seed should be fixed.
    test_data_shuffle_seed : int, optional
        The seed to choose random data in each class for pretrianing's
        testing. A pair of train seed and test seed should be fixed.
    """

    total_num_classes = 100

    # def __init__(
    #     self,
    #     data_dir_path: str,
    #     num_classes_for_pretrain: int,
    #     num_imgs_from_chosen_classes: list,
    #     train: bool,
    #     choosing_class_seed: int = 2022,
    #     train_data_shuffle_seed: int = 222,
    #     test_data_shuffle_seed: int = 223
    # ) -> None:

    #     if train:
    #         dset = 'train'
    #         data_seed = train_data_shuffle_seed
    #     else:
    #         dset = 'test'
    #         data_seed = test_data_shuffle_seed
    #     data_dir_path = os.path.join(data_dir_path, dset)

    #     data = {}
    #     class_list = []
    #     for class_idx, class_data_dir_name in\
    #             enumerate(os.listdir(data_dir_path)):
    #         class_name = class_data_dir_name
    #         class_list.append(class_name)
    #         class_data_dir_path = os.path.join(
    #             data_dir_path,
    #             class_data_dir_name
    #         )
    #         class_image_list = os.listdir(class_data_dir_path)
    #         class_image_path_list = [
    #             os.path.join(
    #                 data_dir_path,
    #                 class_data_dir_name,
    #                 x
    #             ) for x in class_image_list
    #         ]
    #         # We shuffle the data list in this class for fair training
    #         random.seed(data_seed)
    #         random.shuffle(class_image_path_list)
    #         data[class_name] = class_image_path_list

    #     # We randomly pick the classes for fair pretraining.
    #     random.seed(choosing_class_seed)
    #     classes_chosen_for_pretrain = random.sample(
    #         class_list, num_classes_for_pretrain
    #     )
    #     classes_chosen_for_pretrain = sorted(classes_chosen_for_pretrain)

    #     chosen_idx = 0
    #     self.pretrain_data = []
    #     for num_chosen_imgs, num_classes_for_num_imgs in\
    #             num_imgs_from_chosen_classes:
    #         for clas in range(num_classes_for_num_imgs):
    #             chosen_class = classes_chosen_for_pretrain[chosen_idx]
    #             chosen_data = data[chosen_class][-num_chosen_imgs:]
    #             chosen_data = [
    #                 (x, chosen_idx, chosen_class) for x in chosen_data
    #             ]
    #             self.pretrain_data.extend(chosen_data)
    #             chosen_idx += 1

    def read_data(
        self,
        data_dir_path: str,
        train_data_shuffle_seed: int,
        test_data_shuffle_seed: int
    ):
        """read_data _summary_

        _extended_summary_

        Parameters
        ----------
        data_dir_path : str
            _description_
        train_data_shuffle_seed : int
            _description_
        test_data_shuffle_seed : int
            _description_
        """
        def class_names_to_indices(data_dir_path):
            import json
            label_file_path = os.path.join(data_dir_path, 'Labels.json')
            with open(label_file_path, 'r') as f:
                label_file = json.load(f)

            class_list = list(label_file.keys())
            name2index_list = {i: n for i, n in enumerate(class_list)}

            return name2index_list

        self.name2index_list = class_names_to_indices(data_dir_path)
        data = {}
        for dset in ['train', 'test']:
            dset_dir_path = os.path.join(data_dir_path, dset)

            if dset == 'train':
                data_seed = train_data_shuffle_seed
            elif dset == 'test':
                data_seed = test_data_shuffle_seed

            for class_idx in self.name2index_list:
                class_name = self.name2index_list[class_idx]

                class_data_dir_path = os.path.join(dset_dir_path, class_name)
                class_image_list = os.listdir(class_data_dir_path)
                class_image_path_list = [
                    os.path.join(
                        dset_dir_path,
                        class_name,
                        x
                    ) for x in class_image_list
                ]

                random.seed(data_seed)
                random.shuffle(class_image_path_list)

                try:
                    data[class_idx].extend(class_image_path_list)
                except KeyError:
                    data[class_idx] = class_image_path_list

        del class_image_path_list
        return data

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
            (height, width, 3), its class and the class's name
        """
        path, chosen_class = self.chosen_data[idx]
        img = Image.open(path)
        return (img, chosen_class)


if __name__ == '__main__':
    temp = LIGETIPretrainImageNet100(
        data_dir_path='/data/imagenet100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (100, 10), (250, 20), (500, 10)
        ],
        train=True,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222
    )

    temp = LIGETIPretrainImageNet100(
        data_dir_path='/data/imagenet100',
        num_classes_for_pretrain=40,
        num_imgs_from_chosen_classes=[
            (50, 40)
        ],
        train=False,
        choosing_class_seed=2022,
        train_data_shuffle_seed=223,
        test_data_shuffle_seed=222
    )
