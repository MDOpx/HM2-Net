import os
import h5py
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def random_rot_flip(image, label):
    """随机旋转和翻转图像及标签"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    """随机旋转一定角度"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=3)
    label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False, order=0)
    return image, label


class RandomGenerator(object):
    """随机增强生成器"""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 随机旋转或翻转
        if np.random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif np.random.random() > 0.5:
            image, label = random_rotate(image, label)

        # 获取图像尺寸
        shape = image.shape
        if len(shape) == 3:  # RGB 图像
            x, y, c = shape
            zoom_factors = (self.output_size[0] / x, self.output_size[1] / y, 1)
        elif len(shape) == 2:  # 单通道图像
            x, y = shape
            zoom_factors = (self.output_size[0] / x, self.output_size[1] / y)
        else:
            raise ValueError(f"Unsupported image shape: {shape}")

        # 缩放到指定输出大小
        image = zoom(image, zoom_factors, order=3)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 转换为 PyTorch Tensor
        if len(shape) == 3:  # RGB 图像
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # 转换为 (C, H, W)
        else:  # 单通道
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # 增加通道维度

        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample



class CVC_dataset(Dataset):
    """CVC 数据集类，用于训练和验证"""
    def __init__(self, base_dir, list_dir, split, transform=None):
        """
        Args:
            base_dir (str): 数据集根目录，包含 .npz 文件
            list_dir (str): 包含文件名的 .txt 文件目录
            split (str): 数据集划分标识，例如 'train', 'val'
            transform (callable, optional): 数据增强方法
        """
        self.transform = transform
        self.split = split
        list_path = os.path.join(list_dir, f"{split}.txt")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"文件列表 {list_path} 不存在，请检查路径。")
        self.sample_list = open(list_path).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        try:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, f"{slice_name}.npz")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件 {data_path} 不存在。")

            data = np.load(data_path)
            image, label = data['image'], data['label']

            sample = {'image': image, 'label': label}
            if self.transform:
                sample = self.transform(sample)

            sample['case_name'] = slice_name
            return sample
        except Exception as e:
            print(f"加载训练/验证数据时出错：{e}")
            raise e


class CVC_TestDataset(Dataset):
    """CVC 数据集类，适配测试集（只读取 image，没有 label）"""
    def __init__(self, base_dir, list_dir, split, img_size=224):
        """
        Args:
            base_dir (str): 数据集根目录，包含 .npy.h5 文件
            list_dir (str): 包含文件名的 .txt 文件目录
            split (str): 数据集划分标识，例如 'test'
            img_size (int): 模型输入的图像尺寸（默认 224x224）
        """
        self.split = split
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),  # 调整尺寸为 224x224
        ])
        list_path = os.path.join(list_dir, f"{split}.txt")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"文件列表 {list_path} 不存在，请检查路径。")
        self.sample_list = open(list_path).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        try:
            # 获取文件名
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, f"{slice_name}.npy.h5")

            # 检查文件是否存在
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件 {data_path} 不存在。")

            # 读取图像
            with h5py.File(data_path, 'r') as data:
                image = data['image'][:]  # 图像数据

            # 转换为 RGB 格式 (C, H, W)
            if image.ndim == 2:  # 如果是灰度图，扩展为三通道
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:  # 单通道，扩展为三通道
                image = np.concatenate([image] * 3, axis=-1)

            # 转换为 PyTorch Tensor，并调整维度为 (C, H, W)
            image = self.transform(Image.fromarray(image))  # 使用 PIL 转换
            image = image / 255.0  # 归一化到 [0, 1]

            # 返回样本
            sample = {'image': image, 'case_name': slice_name}
            return sample

        except Exception as e:
            print(f"加载数据时出错：{e}")
            raise e

