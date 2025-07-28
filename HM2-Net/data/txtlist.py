import os
import random


def generate_train_val_split(data_dir, output_dir, train_ratio=0.8, seed=1234):
    """
    根据指定的比例划分训练集和验证集，并生成对应的 train.txt 和 val.txt 文件。

    Args:
        data_dir (str): 存储 .npz 文件的数据集路径。
        output_dir (str): 输出 train.txt 和 val.txt 文件的目录。
        train_ratio (float): 训练集所占比例（默认为 0.8，即 80% 用于训练）。
        seed (int): 随机种子，确保划分结果可复现。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .npz 文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    all_files = sorted([os.path.splitext(f)[0] for f in all_files])  # 去掉扩展名

    # 设置随机种子
    random.seed(seed)

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 计算训练集和验证集大小
    total_files = len(all_files)
    train_size = int(total_files * train_ratio)

    # 划分训练集和验证集
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]

    # 输出 train.txt 和 val.txt 文件
    train_txt_path = os.path.join(output_dir, 'train.txt')
    val_txt_path = os.path.join(output_dir, 'val.txt')

    with open(train_txt_path, 'w') as train_file:
        train_file.writelines(f"{line}\n" for line in train_files)

    with open(val_txt_path, 'w') as val_file:
        val_file.writelines(f"{line}\n" for line in val_files)

    print(f"训练集文件已保存到: {train_txt_path}")
    print(f"验证集文件已保存到: {val_txt_path}")



def generate_train_val_split_all(data_dir, output_dir, train_ratio=0.8, seed=1234):
    """
    根据指定的比例划分训练集和验证集，并生成对应的 train.txt 和 val.txt 文件。

    Args:
        data_dir (str): 存储 .npz 文件的数据集路径。
        output_dir (str): 输出 train.txt 和 val.txt 文件的目录。
        train_ratio (float): 训练集所占比例（默认为 0.8，即 80% 用于训练）。
        seed (int): 随机种子，确保划分结果可复现。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .npz 文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    all_files = sorted([os.path.splitext(f)[0] for f in all_files])  # 去掉扩展名

    # 设置随机种子
    random.seed(seed)

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 计算训练集和验证集大小
    total_files = len(all_files)
    train_size = int(total_files * train_ratio * train_ratio)
    val_size = int(total_files * train_ratio)

    # 划分训练集和验证集
    train_files = all_files[:train_size]
    val_files = all_files[train_size:val_size]
    test_files = all_files[val_size:]

    # 输出 train.txt 和 val.txt 文件
    train_txt_path = os.path.join(output_dir, 'train.txt')
    val_txt_path = os.path.join(output_dir, 'val.txt')
    test_txt_path = os.path.join(output_dir, 'test.txt')

    with open(train_txt_path, 'w') as train_file:
        train_file.writelines(f"{line}\n" for line in train_files)

    with open(val_txt_path, 'w') as val_file:
        val_file.writelines(f"{line}\n" for line in val_files)
    with open(test_txt_path, 'w') as test_file:
        test_file.writelines(f"{line}\n" for line in test_files)

    print(f"训练集文件已保存到: {train_txt_path}")
    print(f"验证集文件已保存到: {val_txt_path}")
    print(f"测试集文件已保存到: {test_txt_path}")

def generate_train_val_split_all_png(data_dir, output_dir, train_ratio=0.8, seed=1234):
    """
    根据指定的比例划分训练集和验证集，并生成对应的 train.txt 和 val.txt 文件。

    Args:
        data_dir (str): 存储 .npz 文件的数据集路径。
        output_dir (str): 输出 train.txt 和 val.txt 文件的目录。
        train_ratio (float): 训练集所占比例（默认为 0.8，即 80% 用于训练）。
        seed (int): 随机种子，确保划分结果可复现。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .npz 文件
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    all_files = sorted([os.path.splitext(f)[0] for f in all_files])  # 去掉扩展名

    # 设置随机种子
    random.seed(seed)

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 计算训练集和验证集大小
    total_files = len(all_files)
    train_size = int(total_files * train_ratio * train_ratio)
    val_size = int(total_files * train_ratio)

    # 划分训练集和验证集
    train_files = all_files[:train_size]
    val_files = all_files[train_size:val_size]
    test_files = all_files[val_size:]

    # 输出 train.txt 和 val.txt 文件
    train_txt_path = os.path.join(output_dir, 'train.txt')
    val_txt_path = os.path.join(output_dir, 'val.txt')
    test_txt_path = os.path.join(output_dir, 'test.txt')

    with open(train_txt_path, 'w') as train_file:
        train_file.writelines(f"{line}\n" for line in train_files)

    with open(val_txt_path, 'w') as val_file:
        val_file.writelines(f"{line}\n" for line in val_files)
    with open(test_txt_path, 'w') as test_file:
        test_file.writelines(f"{line}\n" for line in test_files)

    print(f"训练集文件已保存到: {train_txt_path}")
    print(f"验证集文件已保存到: {val_txt_path}")
    print(f"测试集文件已保存到: {test_txt_path}")

# 示例使用
data_dir = "/home/shuoxing/data/TransUNet/data/CTerwo/train/npz"  # 替换为你的 .npz 数据集目录
output_dir = "/home/shuoxing/data/TransUNet/data/CTerwo/train/lists"  # 替换为存储 train.txt 和 val.txt 的目录
generate_train_val_split_all_png(data_dir, output_dir, train_ratio=0.8, seed=1234)
