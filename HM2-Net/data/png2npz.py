import os
import numpy as np
from PIL import Image


def create_npz_dataset(input_image_dir, label_image_dir, output_dir):
    """
    将 RGB 输入图像和单通道标签图像合并为 npz 文件，同时保留原始图像和标签。

    Args:
        input_image_dir (str): 输入图像文件夹路径（RGB图像）。
        label_image_dir (str): 标签图像文件夹路径（单通道灰度图像）。
        output_dir (str): 输出 npz 文件的保存目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入图像和标签图像的文件列表
    input_files = sorted(os.listdir(input_image_dir))
    label_files = sorted(os.listdir(label_image_dir))

    # 检查输入和标签文件数量是否匹配
    if len(input_files) != len(label_files):
        raise ValueError("输入图像和标签图像的数量不匹配！")

    # 遍历文件，生成 npz 数据集
    for input_file, label_file in zip(input_files, label_files):
        # 检查文件名是否一致
        if os.path.splitext(input_file)[0] != os.path.splitext(label_file)[0]:
            raise ValueError(f"文件名不匹配: {input_file} 和 {label_file}")

        # 加载输入图像（RGB）
        input_path = os.path.join(input_image_dir, input_file)
        image = np.array(Image.open(input_path).convert('RGB'))  # 转换为 RGB 格式
        if image.shape[-1] != 3:
            raise ValueError(f"输入图像 {input_file} 不是 3 通道的 RGB 图像")

        # 加载标签图像（单通道）
        label_path = os.path.join(label_image_dir, label_file)
        label = np.array(Image.open(label_path).convert('L'))  # 转换为单通道灰度图
        label[label != 255] = 0
        label[label != 0] = 1

        # 输出 npz 文件路径
        output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}.npz")

        # 保存为 npz 文件（不删除原始文件）
        np.savez_compressed(output_path, image=image, label=label)
        print(f"保存 npz 文件: {output_path}")


# 示例使用
def png2npz_data(dataset_name):
    input_image_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/images'  # 替换为 RGB 图像文件夹路径
    label_image_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/masks'  # 替换为标签图像文件夹路径
    output_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/npz'  # 替换为输出 npz 文件的保存路径

    create_npz_dataset(input_image_dir, label_image_dir, output_dir)

    # input_image_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/test/images'  # 替换为 RGB 图像文件夹路径
    # label_image_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/test/masks'  # 替换为标签图像文件夹路径
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/test/npz'  # 替换为输出 npz 文件的保存路径
    #
    # create_npz_dataset(input_image_dir, label_image_dir, output_dir)

    # input_image_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/val/images'  # 替换为 RGB 图像文件夹路径
    # label_image_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/val/masks'  # 替换为标签图像文件夹路径
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/val/npz'  # 替换为输出 npz 文件的保存路径
    #
    # create_npz_dataset(input_image_dir, label_image_dir, output_dir)

    # input_image_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/images'  # 替换为 RGB 图像文件夹路径
    # label_image_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/masks'  # 替换为标签图像文件夹路径
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/npz'  # 替换为输出 npz 文件的保存路径
    #
    # create_npz_dataset(input_image_dir, label_image_dir, output_dir)

# dataset_name = 'CVC-300'
# png2npz_data(dataset_name)
# dataset_name = 'CVC-ClinicDB'
# png2npz_data(dataset_name)
# dataset_name = 'CVC-ColonDB'
# png2npz_data(dataset_name)
# dataset_name = 'ETIS-LaribPolypDB'
# png2npz_data(dataset_name)
# dataset_name = 'Kvasir'
# png2npz_data(dataset_name)

# dataset_name = 'UDIAT'
# png2npz_data(dataset_name)
dataset_name = 'CTerwo'
png2npz_data(dataset_name)
