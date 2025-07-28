import os
import h5py
import numpy as np
from PIL import Image


def convert_and_generate_test_txt(input_folder, output_folder, txt_folder, txt_name):
    """
    将 .png 文件转换为 .npy.h5 格式，同时生成包含文件名的 test.txt 文件。
    Args:
        input_folder (str): 包含测试 .png 文件的文件夹路径。
        output_folder (str): 保存 .npy.h5 文件的输出文件夹路径。
        txt_folder (str): 保存 test.txt 文件的文件夹路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    file_names = []

    # 遍历 .png 文件并转换为 .npy.h5
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # 加载 .png 文件
            file_path = os.path.join(input_folder, filename)
            image = Image.open(file_path).convert('RGB')  # 转换为 RGB 格式
            image_array = np.array(image, dtype=np.uint8)  # 转为 numpy 数组

            # 保存为 .npy.h5 文件
            base_name = os.path.splitext(filename)[0]
            h5_path = os.path.join(output_folder, f"{base_name}.npy.h5")
            with h5py.File(h5_path, 'w') as h5_file:
                h5_file.create_dataset('image', data=image_array, dtype='uint8')  # 保存图像数据
            print(f"已转换并保存: {h5_path}")

            # 记录文件名（不含扩展名）
            file_names.append(base_name)

    # 生成 test.txt 文件
    test_txt_path = os.path.join(txt_folder, txt_name)
    with open(test_txt_path, 'w') as f:
        for name in file_names:
            f.write(name + '\n')

    print(f"测试集文件已生成: {test_txt_path}")


def png2h5_data(dataset_name):
    # 输入文件夹路径（.png 文件所在目录）
    input_folder = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/images'  # 替换为你的 .png 文件夹路径
    # 输出文件夹路径（.npy.h5 保存目录）
    output_folder = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/h5'  # 替换为保存 .npy.h5 文件的路径
    # 保存 .txt 文件的路径
    txt_folder = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/lists'  # 替换为保存 .txt 文件的路径
    # 执行转换和生成 test.txt 文件
    convert_and_generate_test_txt(input_folder, output_folder, txt_folder,'test.txt')

    # # 输入文件夹路径（.png 文件所在目录）
    # input_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/images'  # 替换为你的 .png 文件夹路径
    # # 输出文件夹路径（.npy.h5 保存目录）
    # output_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/h5'  # 替换为保存 .npy.h5 文件的路径
    # # 保存 .txt 文件的路径
    # txt_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/lists'  # 替换为保存 .txt 文件的路径
    # # 执行转换和生成 test.txt 文件
    # convert_and_generate_test_txt(input_folder, output_folder, txt_folder, 'train.txt')


if __name__ == "__main__":
    # dataset_name = 'CVC-300'
    # png2h5_data(dataset_name)
    # dataset_name = 'CVC-ClinicDB'
    # png2h5_data(dataset_name)
    # dataset_name = 'CVC-ColonDB'
    # png2h5_data(dataset_name)
    # dataset_name = 'ETIS-LaribPolypDB'
    # png2h5_data(dataset_name)
    # dataset_name = 'Kvasir'
    # png2h5_data(dataset_name)
    # dataset_name = 'UDIAT'
    # png2h5_data(dataset_name)
    # dataset_name = 'BUSI'
    # png2h5_data(dataset_name)
    dataset_name = 'CTerwo'
    png2h5_data(dataset_name)



