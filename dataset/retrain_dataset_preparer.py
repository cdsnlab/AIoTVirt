'''
Author: Tung Nguyen (tungnt@kaist.ac.kr)
'''

import os
import sys
import random
from pickle import load
from typing import Tuple
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)
sys.path.append(BASE_DIR)    


class RetrainingDatasetPreparer(object):
    """RetrainingDatasetPreparer Split a dataset into several subsets for
    retraining, each of which is corresponding to a retraining task.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset. For example, cifar10, imagenet, cifar100,
        flower102...
    data_dir_path : str
        Path to the dataset
    num_total_classes : int
        Number of total classes within the dataset
    num_test_images_each_class : int
        Number of images used as test images for from each of the classes
        in the dataset.
    num_total_images_each_task : int
        Number of images used for each retraining task.
    task_specifications : list
        A list of specifications for retraining. Each element from the list
        specifies a training task. An element is a tuple containing a few
        important variables. (1) The number of new classes for class
        incremental learning. (2) The number of the old classes that need to
        be retrained. (3) The seed to randomly determine the number of images
        in each class.
    choosing_class_seed : int, optional
        This determines the classes that will be used for retrianing. Must be
        identical to the one we used for pretraining.
        by default 2022
    retrain_data_shuffle_seed : int, optional
        Data from each retraining task will be shuffled using this seed
        by default 2
    pretrain_test_data_shuffle_seed : int, optional
        Data to be used as test data throughout the training will be determined
        by this seed, by default 222
    """
    def __init__(
        self,
        dataset_name: str,
        data_dir_path: str,
        num_total_classes: int,
        num_pretrain_classes: int,
        num_test_images_each_class: int,
        num_total_images_each_task: int,
        task_specifications: list,
        choosing_class_seed: int = 2022,
        retrain_data_shuffle_seed: int = 2,
        pretrain_test_data_shuffle_seed: int = 222
    ):
        self.data_dir_path = data_dir_path
        self.num_total_classes = num_total_classes
        self.retrain_data_shuffle_seed = retrain_data_shuffle_seed
        self.pretrain_test_data_shuffle_seed = pretrain_test_data_shuffle_seed
        self.num_total_images_each_task = num_total_images_each_task
        self.num_test_images_each_class = num_test_images_each_class

        # Because of the data format each dataset will require a bit different
        # function to load them into a nice and tight list.
        if 'cifar' in dataset_name:
            self.total_retrain_data, self.class_list = self.cifar_reader()

            # Determine that classes that have been used for PREtraining
            # Once a again, go back and check if the choosing_class_seed here is
            # identical to the one used for pretraining.
            # If they dont match, we have trouble.
            random.seed(choosing_class_seed)
            pretrain_classes = sorted(
                random.sample(
                    sorted(self.class_list), num_pretrain_classes
                )
            )
        elif 'imagenet' in dataset_name:
            self.total_retrain_data, self.class_list = self.imagenet_reader()
            random.seed(choosing_class_seed)
            pretrain_classes = sorted(
                random.sample(
                    self.class_list, num_pretrain_classes
                )
            )

        self.seen_classes = pretrain_classes.copy()
        # print(self.seen_classes)

        # Retrain/unseen classes are all the classes in the dataset except
        # the ones we have pretrained models with.
        self.retrain_classes = self.class_list.copy()
        for cls in pretrain_classes:
            self.retrain_classes.remove(cls)
        self.unseen_classes = self.retrain_classes.copy()

        # [task1's data, task2's data]
        self.all_task_retrain_data = []

        for specs in task_specifications:
            classes_for_retrain, nums_images_of_classes = \
                self.translate_task_specs_to_data(specs)

            # The first element of each tuple is the number of new classes
            # If this number is not 0, then we are during class-incremental
            # learning
            is_incremental_learning = False
            if specs[0] != 0:
                is_incremental_learning = True
            task_pretrain_data = self.extract_data_for_retrain_task(
                classes_for_retrain,
                nums_images_of_classes,
                is_incremental_learning
            )
            self.all_task_retrain_data.append(task_pretrain_data)

    def translate_task_specs_to_data(self, specs: tuple):
        """translate_task_specs_to_data Translate specs into the specific
        number of images for each individual class.


        Parameters
        ----------
        specs : tuple
            Contains (1) number of new classes (2) number of old classes and
            (3) a seed to determine the number of each images for each class
            during this retraining task

        Returns
        -------
        list, list
            the classes that will be used for retraining
            and the numbers of images for those classes.
        """
        # num_new_classes determines how many new classes will be
        # incrementally learned during this task
        # num_old_classes determines how many old classes from
        # which data to be drawn to perform data-incremental learning
        num_new_classes, num_old_classes, seed = specs
        if num_new_classes and not num_old_classes:
            nums_images_of_classes = \
                [self.num_total_images_each_task // num_new_classes for _ in
                    range(num_new_classes)]
            classes_for_retrain = self.unseen_classes[:num_new_classes]
        elif not num_new_classes and num_old_classes:
            nums_images_of_classes = self.match_class_data_to_total_data(
                num_old_classes,
                seed=seed
                )
            classes_for_retrain = self.seen_classes

        return classes_for_retrain, nums_images_of_classes

    def extract_data_for_retrain_task(
        self,
        classes_for_retrain: list,
        nums_images_of_classes: list,
        is_incremental_learning: bool = True,
    ):
        print(self.seen_classes)
        """extract_data_for_retrain_task Extract data for the retraining task

        Parameters
        ----------
        nums_images_of_classes : list
            the numbers of images for classes that will be used for retraining
        """
        task_retrain_data = []
        for class_idx, (class_name, num_images) in enumerate(
                zip(classes_for_retrain, nums_images_of_classes)):
            class_data = self.total_retrain_data[class_name][:num_images]
            self.total_retrain_data[class_name] = \
                self.total_retrain_data[class_name][num_images:]
            if is_incremental_learning:
                self.unseen_classes.remove(class_name)
                self.seen_classes.append(class_name)

            class_data = [
                (x, self.seen_classes.index(class_name), class_name)
                for x in class_data
            ]
            task_retrain_data.extend(class_data)

        random.seed(self.retrain_data_shuffle_seed)
        random.shuffle(task_retrain_data)

        return task_retrain_data

    def match_class_data_to_total_data(
        self,
        num_classes: int,
        seed: int,
        num_images_each_class_thrs_low: float = 0.8,
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
        """
        avg_num_each_class = self.num_total_images_each_task // num_classes
        nums_images_of_classes = []
        random.seed(seed)
        for _ in range(int(num_classes/2)):
            num_image_this_class = random.randint(
                int(avg_num_each_class*num_images_each_class_thrs_low),
                avg_num_each_class
            )
            nums_images_of_classes.append(num_image_this_class)
        for _ in range(int(num_classes/2)+1, num_classes):
            num_image_this_class = random.randint(
                avg_num_each_class,
                int(avg_num_each_class*num_images_each_class_thrs_high),
            )
            nums_images_of_classes.append(num_image_this_class)
        nums_images_of_classes.append(self.num_total_images_each_task -
                                      sum(nums_images_of_classes))

        return nums_images_of_classes

    def cifar_reader(self):
        data = {}
        class_list = []
        for class_data_file_name in os.listdir(self.data_dir_path):
            class_idx = int(class_data_file_name.split('.')[0])
            class_list.append(class_idx)
            class_data_file_path = os.path.join(
                self.data_dir_path,
                class_data_file_name
            )
            with open(class_data_file_path, 'rb') as f:
                class_data = load(f)
                f.close()
                random.seed(self.pretrain_test_data_shuffle_seed)
                random.shuffle(class_data)
                data[class_idx] = class_data[self.num_test_images_each_class:]
        return data, class_list

    def imagenet_reader(self):
        data = {}
        class_list = []
        for class_data_dir_name in os.listdir(self.data_dir_path):
            class_name = class_data_dir_name
            class_list.append(class_name)
            class_data_dir_path = os.path.join(
                self.data_dir_path,
                class_data_dir_name
            )
            class_image_list = os.listdir(class_data_dir_path)
            class_image_path_list = [
                os.path.join(
                    self.data_dir_path,
                    class_data_dir_name,
                    x
                ) for x in class_image_list
            ]
            random.seed(self.pretrain_test_data_shuffle_seed)
            random.shuffle(class_image_path_list)
            data[class_name] = \
                class_image_path_list[self.num_test_images_each_class:]
        return data, class_list


class ImagenetRetrainDataset(object):
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
        pretrain_test_data_shuffle_seed: int = 222
    ):
        self.task_training_dataset = RetrainingDatasetPreparer(
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
        # print(len(self.task_training_dataset))

    def __call__(self, idx):
        return self.task_training_dataset[idx]


class CIFARRetrainDataset(object):
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
        pretrain_test_data_shuffle_seed: int = 222
    ):
        self.task_training_dataset = RetrainingDatasetPreparer(
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
        # print(len(self.task_training_dataset))

    def __call__(self, idx):
        return self.task_training_dataset[idx]


if __name__ == '__main__':
    dataset = 'cifar10'
    dataset = CIFARRetrainDataset(
        task_num=4,
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
    print(dataset(10))
