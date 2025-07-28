import os
import numpy as np

def binarize_label(label):
    """
    对标签进行二值化处理，将灰度值大于0的部分设为1，其余设为0。
    Args:
        label (np.ndarray): 原始标签数组
    Returns:
        np.ndarray: 二值化后的标签数组
    """
    return (label > 0).astype(np.uint8)

def process_npz_folder(input_folder):
    """
    遍历文件夹中的所有 .npz 文件，对标签进行二值化并覆盖保存。
    Args:
        input_folder (str): 输入 .npz 文件所在的文件夹
    """
    for filename in os.listdir(input_folder):
        if filename.endswith('.npz'):
            file_path = os.path.join(input_folder, filename)

            # 加载 .npz 文件
            data = np.load(file_path)
            image = data['image']
            label = data['label']

            # 对标签进行二值化
            binarized_label = binarize_label(label)

            # 直接覆盖保存
            np.savez_compressed(file_path, image=image, label=binarized_label)
            print(f"已处理并覆盖: {filename}")

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "/home/shuoxing/data/TransUNet/data_finals/ETIS-LaribPolypDB/test/npz"  # 替换为你的实际路径

    process_npz_folder(input_folder)
