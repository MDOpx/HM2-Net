from PIL import Image
import os
import shutil
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def interactive_threshold_adjuster(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("Error: 图像加载失败")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建图形界面
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    # 显示原始图像
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 初始阈值处理
    threshold = 128
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img_display = ax2.imshow(thresholded, cmap='gray')
    ax2.set_title('Thresholded Image')
    ax2.axis('off')

    # 添加阈值滑块
    ax_threshold = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_threshold, 'Threshold', 0, 255, valinit=threshold)

    def update(val):
        threshold = int(slider.val)
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        img_display.set_data(thresholded)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def process_pixels(image_path):
    # 加载图像并转换为数组
    img = Image.open(image_path)
    img_array = np.array(img)

    # 获取唯一像素值及其频次
    unique_vals, counts = np.unique(img_array, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]

    # 可视化像素分布
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.bar(unique_vals[sorted_idx], counts[sorted_idx])
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')

    # 阈值处理（小于128置零）
    thresholded = np.where(img_array < 128, 0, img_array)
    plt.figure()
    plt.imshow(thresholded, cmap='gray' if len(img_array.shape) == 2 else None)
    plt.title('Thresholded Image (values <128 set to 0)')
    plt.axis('off')

    plt.show()
    return thresholded

def resize_to_match(source_path, target_path, output_path):
    # 打开目标图像获取尺寸
    with Image.open(target_path) as target_img:
        target_size = target_img.size

    # 打开源图像并调整尺寸
    with Image.open(source_path) as source_img:
        resized_img = source_img.resize(target_size, Image.Resampling.LANCZOS)
        resized_img.save(output_path)
        # 调整尺寸
        resized_img = source_img.resize(target_size, Image.Resampling.LANCZOS)

        # 转换为numpy数组进行处理
        img_array = np.array(resized_img)
        target_value = 90
        # 将指定像素值置零
        if len(img_array.shape) == 3:  # 彩色图像
            mask = (img_array[:, :, 0] <= target_value) & \
                   (img_array[:, :, 1] <= target_value) & \
                   (img_array[:, :, 2] <= target_value)
            img_array[mask] = 0
        else:  # 灰度图像
            img_array[img_array <= target_value] = 0

        # 保存结果
        Image.fromarray(img_array).save(output_path)



# dataset = 'CVC-300'
def resize_dataset(dataset):
    image_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/unmap_'+dataset
    target_folder = '/home/shuoxing/data/TransUNet/data/' + dataset + '/test/images'
    output_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/reunmap_'+dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        img_name = os.path.join(image_folder,filename)
        target_name = os.path.join(target_folder,filename)
        output_name = os.path.join(output_folder,filename)
        if filename.endswith('.png'):
            resize_to_match(img_name, target_name, output_name)


def txt2dir_data(dataset_name,class_name):
    input_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/reunmap_' + dataset_name
    input_txt = '/home/shuoxing/data/TransUNet/data/' + dataset_name + '/train/lists/'+class_name+'.txt'
    output_dir = '/home/shuoxing/data/TransUNet/Results-Final-MFC/unmapdir_' + dataset_name + '/'+class_name

    if not os.path.exists(output_dir.replace('/'+class_name,'')):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir)
        print(f"文件夹 {output_dir} 已创建")
    if not os.path.exists(output_dir):
        # 创建文件夹（包括必要的父目录）
        os.makedirs(output_dir)
        print(f"文件夹 {output_dir} 已创建")

    with open(input_txt, 'r') as f:
        file_names = [line.strip() for line in f]  # 使用strip()移除每行末尾的换行符

    for file_name in file_names:
        in_file = os.path.join(input_dir, file_name+'.png')
        print(in_file)
        out_file = os.path.join(output_dir, file_name+'.png')
        shutil.move(in_file, out_file)


# 使用示例
# interactive_threshold_adjuster('/home/shuoxing/data/TransUNet/Results-Final-MFC/unmap_CVC-ClinicDB/1.png')
# 使用示例
# processed = process_pixels('/home/shuoxing/data/TransUNet/Results-Final-MFC/unmap_CVC-ClinicDB/1.png')
# Image.fromarray(processed).save('output.jpg')

# resize_dataset('CVC-300')
resize_dataset('CVC-ClinicDB')
resize_dataset('CVC-ColonDB')
resize_dataset('ETIS-LaribPolypDB')
resize_dataset('Kvasir')
resize_dataset('BUSI')
resize_dataset('UDIAT')

# class_name = 'test'
#
# # dataset_name = 'CVC-300'
# # txt2dir_data(dataset_name,class_name)
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
#
# class_name = 'val'
#
# # dataset_name = 'CVC-300'
# # txt2dir_data(dataset_name,class_name)
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
#
# class_name = 'train'
#
# # dataset_name = 'CVC-300'
# # txt2dir_data(dataset_name,class_name)
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


