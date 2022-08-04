import os
import sys
import os.path
import pickle
from torchvision.datasets.utils import check_integrity,\
    download_and_extract_archive

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)


class CIFAR10(object):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    num_classes = 10

    def __init__(
            self,
            root: str,
            train: str = True,
            download: bool = False,
    ):
        self.root = root
        self.train = train
        num_classes = self.num_classes
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
            num_imgs = 50000
            dset = 'train'
        else:
            downloaded_list = self.test_list
            num_imgs = 10000
            dset = 'test'

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                # entry['data'] = entry['data'].tolist()
                self.data.extend(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        os.makedirs(
            './dataloader/cifar{}/{}'.format(num_classes, dset),
            exist_ok=True
        )
        self.data = [x for y, x in sorted(
            zip(self.targets, self.data),
            key=lambda x: x[0]
        )]
        for i in range(num_imgs):
            self.data[i] = self.data[i].reshape(3, 32, 32)
            
        self.new_data = {}
        num_each_class_imgs = num_imgs // num_classes
        for c in range(num_classes):
            self.new_data[c] = self.data[
                c*num_each_class_imgs: (c+1)*num_each_class_imgs
            ]
            with open('./dataloader/cifar{}/{}/{}.pkl'.format(
                num_classes,
                dset,
                c
            ), 'wb') as f:
                pickle.dump(self.new_data[c], f)
            f.close()

        # self.targets = sorted(self.targets)
        # print(self.data[0], self.targets[0])
        # print(self.data[1], self.targets[1])

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    num_classes = 100


if __name__ == '__main__':
    dataset = CIFAR100(
        root='./dataloader/',
        train=True,
        download=True
    )