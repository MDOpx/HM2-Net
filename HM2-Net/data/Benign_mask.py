import os
import shutil

def find_and_copy_files(source_dir, target_dir, output_dir):
    """
    在 source_dir 中查找与 target_dir 中文件名相同的文件，并将它们复制到 output_dir。

    参数:
        source_dir (str): 源文件夹路径。
        target_dir (str): 目标文件夹路径。
        output_dir (str): 输出文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取 target_dir 中所有文件的文件名
    target_files = set(os.listdir(target_dir))

    # 遍历 source_dir 中的文件
    for filename in os.listdir(source_dir):
        # 如果文件名在 target_files 中
        if filename in target_files:
            # 构造源文件路径和目标文件路径
            source_file_path = os.path.join(source_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            # 复制文件
            shutil.copy2(source_file_path, output_file_path)
            print(f"Copied {filename} to {output_dir}")

# 示例使用
if __name__ == "__main__":
    source_dir = r"D:\619\UDIAT\all_masks"
    target_dir = r"D:\619\UDIAT\Benign"
    output_dir = r"D:\619\UDIAT\Benign_mask"
    find_and_copy_files(source_dir, target_dir, output_dir)