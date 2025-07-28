import os
import shutil
import h5py
import numpy as np
from PIL import Image


def txt2dir_data(dataset_name,class_name):
    input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/images'
    input_txt = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/lists/'+class_name+'.txt'
    output_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/images'

    if not os.path.exists(output_dir.replace('/images','')):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir)
        print(f"文件夹 {output_dir} 已创建")

    if not os.path.exists(output_dir):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir)
        print(f"文件夹 {output_dir} 已创建")

    if not os.path.exists(output_dir.replace('images','h5')):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir.replace('images','h5'))
        print(f"文件夹 {output_dir.replace('images','h5')} 已创建")

    if not os.path.exists(output_dir.replace('images','lists')):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir.replace('images','lists'))
        print(f"文件夹 {output_dir.replace('images','lists')} 已创建")

    if not os.path.exists(output_dir.replace('images','masks')):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir.replace('images','masks'))
        print(f"文件夹 {output_dir.replace('images','masks')} 已创建")

    if not os.path.exists(output_dir.replace('images','npz')):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir.replace('images','npz'))
        print(f"文件夹 {output_dir.replace('images','npz')} 已创建")


    with open(input_txt, 'r') as f:
        file_names = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符

    # with open(input_txt.replace('test','train'), 'r') as f:
    #     file_names_train = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符
    #
    # with open(input_txt.replace('test','val'), 'r') as f:
    #     file_names_val = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符

    # file_names = list(filter(lambda x: x not in file_names_train, file_names))
    # file_names = list(filter(lambda x: x not in file_names_val, file_names))

    for file_name in file_names:
        in_file = os.path.join(input_dir, file_name+'.png')
        print(in_file)
        out_file = os.path.join(output_dir, file_name+'.png')
        shutil.copy(in_file, out_file)


    input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/masks'
    output_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/masks'
    for file_name in file_names:
        in_file = os.path.join(input_dir, file_name+'.png')
        print(in_file)
        out_file = os.path.join(output_dir, file_name+'.png')
        shutil.copy(in_file, out_file)

    # input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/images'
    # input_txt = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/lists/train.txt'
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/images'
    #
    # if not os.path.exists(output_dir.replace('/images','')):
    #     # 创建文件夹（包括必要的父目录）
    #     os.makedirs(output_dir)
    #     print(f"文件夹 {output_dir} 已创建")
    #
    # if not os.path.exists(output_dir):
    #     # 创建文件夹（包括必要的父目录）
    #     os.makedirs(output_dir)
    #     print(f"文件夹 {output_dir} 已创建")
    #
    # if not os.path.exists(output_dir.replace('images','masks')):
    #     # 创建文件夹（包括必要的父目录）
    #     os.makedirs(output_dir.replace('images','masks'))
    #     print(f"文件夹 {output_dir.replace('images','masks')} 已创建")
    #
    # with open(input_txt, 'r') as f:
    #     file_names = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符
    #
    # for file_name in file_names:
    #     in_file = os.path.join(input_dir, file_name+'.png')
    #     print(in_file)
    #     out_file = os.path.join(output_dir, file_name+'.png')
    #     shutil.copy(in_file, out_file)
    #
    #
    # input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/masks'
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/masks'
    # for file_name in file_names:
    #     in_file = os.path.join(input_dir, file_name+'.png')
    #     print(in_file)
    #     out_file = os.path.join(output_dir, file_name+'.png')
    #     shutil.copy(in_file, out_file)
    #
    # input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/images'
    # input_txt = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/lists/val.txt'
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/images'
    #
    # if not os.path.exists(output_dir.replace('/images','')):
    #     # 创建文件夹（包括必要的父目录）
    #     os.makedirs(output_dir)
    #     print(f"文件夹 {output_dir} 已创建")
    #
    # if not os.path.exists(output_dir):
    #     # 创建文件夹（包括必要的父目录）
    #     os.makedirs(output_dir)
    #     print(f"文件夹 {output_dir} 已创建")
    #
    # if not os.path.exists(output_dir.replace('images','masks')):
    #     # 创建文件夹（包括必要的父目录）
    #     os.makedirs(output_dir.replace('images','masks'))
    #     print(f"文件夹 {output_dir.replace('images','masks')} 已创建")
    #
    # with open(input_txt, 'r') as f:
    #     file_names = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符
    #
    # for file_name in file_names:
    #     in_file = os.path.join(input_dir, file_name+'.png')
    #     print(in_file)
    #     out_file = os.path.join(output_dir, file_name+'.png')
    #     shutil.copy(in_file, out_file)
    #
    #
    # input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/masks'
    # output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/xirou/' + dataset_name + '/train/masks'
    # for file_name in file_names:
    #     in_file = os.path.join(input_dir, file_name+'.png')
    #     print(in_file)
    #     out_file = os.path.join(output_dir, file_name+'.png')
    #     shutil.copy(in_file, out_file)

class_name = 'test'

# dataset_name = 'CVC-300'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CVC-ClinicDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CVC-ColonDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'ETIS-LaribPolypDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'Kvasir'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'UDIAT'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'BUSI'
# txt2dir_data(dataset_name,class_name)
dataset_name = 'CTerwo'
txt2dir_data(dataset_name,class_name)

# class_name = 'val'
#
# dataset_name = 'CVC-300'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CVC-ClinicDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CVC-ColonDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'ETIS-LaribPolypDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'Kvasir'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'UDIAT'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'BUSI'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CTerwo'
# txt2dir_data(dataset_name,class_name)

# class_name = 'train'
#
# dataset_name = 'CVC-300'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CVC-ClinicDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CVC-ColonDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'ETIS-LaribPolypDB'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'Kvasir'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'UDIAT'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'BUSI'
# txt2dir_data(dataset_name,class_name)
# dataset_name = 'CTerwo'
# txt2dir_data(dataset_name,class_name)


#
# input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/images'
# input_txt = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/lists/val.txt'
# output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/' + dataset_name + '/val/images'
#
# if not os.path.exists(output_dir.replace('/images','')):
#     # 创建文件夹（包括必要的父目录）
#     os.makedirs(output_dir)
#     print(f"文件夹 {output_dir} 已创建")
#
# if not os.path.exists(output_dir):
#     # 创建文件夹（包括必要的父目录）
#     os.makedirs(output_dir)
#     print(f"文件夹 {output_dir} 已创建")
#
# if not os.path.exists(output_dir.replace('images','masks')):
#     # 创建文件夹（包括必要的父目录）
#     os.makedirs(output_dir.replace('images','masks'))
#     print(f"文件夹 {output_dir.replace('images','masks')} 已创建")
#
# with open(input_txt, 'r') as f:
#     file_names = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符
#
# for file_name in file_names:
#     in_file = os.path.join(input_dir, file_name+'.png')
#     print(in_file)
#     out_file = os.path.join(output_dir, file_name+'.png')
#     shutil.copy(in_file, out_file)
#
#
# input_dir = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/masks'
# output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/' + dataset_name + '/val/masks'
# for file_name in file_names:
#     in_file = os.path.join(input_dir, file_name+'.png')
#     print(in_file)
#     out_file = os.path.join(output_dir, file_name+'.png')
#     shutil.copy(in_file, out_file)


