import numpy as np


def read_npz_info(filepath):
    """
    读取 .npz 文件并打印内容信息，同时判断图像和标签的通道数，并检查标签值范围。
    """
    try:
        # 加载 .npz 文件
        data = np.load(filepath)

        print(f"文件路径: {filepath}")
        print(f"包含以下键: {list(data.keys())}")

        # 遍历所有键并打印信息
        for key in data.keys():
            array = data[key]
            print(f"\n键: {key}")
            print(f"数据形状: {array.shape}")
            print(f"数据类型: {array.dtype}")

            # 判断通道数
            if len(array.shape) == 2:
                print("通道数: 1（单通道，灰度图）")
            elif len(array.shape) == 3:
                print(f"通道数: {array.shape[-1]}（多通道，例如 RGB）")
            else:
                print("通道数: 无法确定，数据可能是高维结构")

            # 如果是标签数据，检查值范围
            if key.lower() == 'label':  # 假设标签键名为 'label'
                unique_values = np.unique(array)
                print(f"标签的唯一值: {unique_values}")
                print(f"标签的最小值: {array.min()}, 最大值: {array.max()}")

        return data

    except Exception as e:
        print(f"读取文件失败: {e}")


# 示例使用
if __name__ == "__main__":
    npz_filepath = "/home/shuoxing/data/TransUNet/data/CVC-ClinicDB/test/npz/10.npz"  # 替换为你的文件路径
    read_npz_info(npz_filepath)
