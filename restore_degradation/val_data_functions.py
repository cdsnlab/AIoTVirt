import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

from dataset_utils import get_dataset


# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir,val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('synthetic','gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(self.val_data_dir + input_name)
        gt_img = Image.open(self.val_data_dir + gt_name)

        # Resizing image in the multiple of 16"
        wd_new,ht_new = input_img.size
        if ht_new>wd_new and ht_new>1024:
            wd_new = int(np.ceil(wd_new*1024/ht_new))
            ht_new = 1024
        elif ht_new<=wd_new and wd_new>1024:
            ht_new = int(np.ceil(ht_new*1024/wd_new))
            wd_new = 1024
        wd_new = int(16*np.ceil(wd_new/16.0))
        ht_new = int(16*np.ceil(ht_new/16.0))
        input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

# --- Validation/test dataset --- #
class ValDataMetaIR(data.Dataset):
    def __init__(self, task: str, crop_size ,is_val=True):
        super().__init__()

        self.crop_size = crop_size
        input_names, gt_names = get_dataset('val', task)
        if is_val==False:
            input_names, gt_names = get_dataset('test', task)

        self.input_names = input_names
        self.gt_names = gt_names
        # self.val_data_dir = val_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        input_img = Image.open(input_name)
        gt_img = Image.open(gt_name)

        if input_img.mode != 'RGB':
            input_img = input_img.convert('RGB')
        if gt_img.mode != 'RGB':
            gt_img = gt_img.convert('RGB')

        
        
        width, height = input_img.size

        if width < crop_width and height < crop_height :
            input_img = input_img.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            input_img = input_img.resize((crop_width,height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            input_img = input_img.resize((width,crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size

        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        x, y = width / 2 - crop_width / 2, height / 2 - crop_height / 2
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)