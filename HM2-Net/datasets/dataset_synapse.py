import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 🔥 调试：打印 image 维度
        print(f"🔥 Debug: image shape before processing: {image.shape}")

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # **🚀 修复 shape 解析问题**
        if len(image.shape) == 3:  # 如果是 (H, W, C)
            x, y, c = image.shape
            scale = (self.output_size[0] / x, self.output_size[1] / y, 1)  # **三维缩放**
            image = zoom(image, scale, order=3)
            image = np.transpose(image, (2, 0, 1))  # **转换为 (C, H, W)**
        elif len(image.shape) == 2:  # 如果是 (H, W)
            x, y = image.shape
            scale = (self.output_size[0] / x, self.output_size[1] / y)  # **二维缩放**
            image = zoom(image, scale, order=3)
            image = np.expand_dims(image, axis=0)  # **确保变成 (1, H, W)**

        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 🔥 确保变成 PyTorch 需要的格式
        image = torch.from_numpy(image.astype(np.float32))  # **(C, H, W)**
        label = torch.from_numpy(label.astype(np.float32))

        print(f"🔥 Debug: image shape after transform: {image.shape}")  # ✅ 应该是 (C, H, W)

        sample = {'image': image, 'label': label.long()}
        return sample




class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
