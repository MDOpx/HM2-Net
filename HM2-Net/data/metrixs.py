import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize

import math


def load_png_files(folder_path):
    """
    加载指定文件夹中的 PNG 文件，并返回文件名列表和对应的图像数据。
    Args:
        folder_path (str): 文件夹路径
    Returns:
        dict: 文件名到图像数组的映射
    """
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图
            data[filename] = image
    return data

def load_png_files_255(folder_path):
    """
    加载指定文件夹中的 PNG 文件，并返回文件名列表和对应的图像数据。
    Args:
        folder_path (str): 文件夹路径
    Returns:
        dict: 文件名到图像数组的映射
    """
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图
            image[image == 255] = 1
            data[filename] = image
    return data

def load_jpg_files_255(folder_path):
    """
    加载指定文件夹中的 PNG 文件，并返回文件名列表和对应的图像数据。
    Args:
        folder_path (str): 文件夹路径
    Returns:
        dict: 文件名到图像数组的映射
    """
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图
            image[image == 255] = 1
            data[filename.replace('.jpg','.png')] = image
    return data


def binarize_image(image):
    """
    将图像进行二值化处理，大于 0 的像素值转化为 1。
    Args:
        image (numpy.ndarray): 输入图像数组。
    Returns:
        numpy.ndarray: 二值化后的图像。
    """
    return (image > 0).astype(np.uint8)


def resize_to_match(image, target_shape):
    """
    调整图像的尺寸以匹配目标形状。
    Args:
        image (numpy.ndarray): 输入图像。
        target_shape (tuple): 目标形状 (height, width)。
    Returns:
        numpy.ndarray: 调整大小后的图像。
    """
    resized_image = resize(image, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    return resized_image


def calculate_metrics(predictions, labels, num_classes=2):
    """
    计算预测结果和标签的定量指标。
    Args:
        predictions (numpy.ndarray): 预测结果数组。
        labels (numpy.ndarray): 标签数组。
        num_classes (int): 分割任务的类别数（包括背景）。
    Returns:
        dict: 包含每个类别的 Dice、IoU、Recall、Precision 和 Hausdorff 距离的字典，以及它们的平均值。
    """
    metrics = {
        "Dice": [],
        "IoU": [],
        "Recall": [],
        "Precision": [],
        "Hausdorff": []
    }

    for cls in range(1, num_classes):  # 从 1 开始计算，跳过背景类
        cls_preds = (predictions == cls).astype(int)
        cls_labels = (labels == cls).astype(int)

        # 计算 Dice 系数
        intersection = np.sum(cls_preds * cls_labels)
        dice = (2. * intersection) / (np.sum(cls_preds) + np.sum(cls_labels) + 1e-6)
        metrics["Dice"].append(dice)

        # 计算 IoU
        union = np.sum(cls_preds) + np.sum(cls_labels) - intersection
        iou = intersection / (union + 1e-6)
        metrics["IoU"].append(iou)

        # 计算 Recall 和 Precision
        recall = intersection / (np.sum(cls_labels) + 1e-6)
        precision = intersection / (np.sum(cls_preds) + 1e-6)
        metrics["Recall"].append(recall)
        metrics["Precision"].append(precision)

        # 计算 Hausdorff 距离
        if cls_preds.sum() > 0 and cls_labels.sum() > 0:  # 确保类存在
            hausdorff = max(
                directed_hausdorff(cls_preds, cls_labels)[0],
                directed_hausdorff(cls_labels, cls_preds)[0]
            )
        else:
            hausdorff = np.inf  # 如果类不存在，则设为无穷大
        metrics["Hausdorff"].append(hausdorff)

    # 计算平均值
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    metrics["Average"] = avg_metrics

    return metrics


def evaluate_folder(prediction_folder, label_folder, output_file, num_classes=2):
    """
    评估两个文件夹中的 PNG 文件，并计算定量指标，同时将结果保存到指定位置。
    Args:
        prediction_folder (str): 存储预测结果的文件夹路径。
        label_folder (str): 存储标签的文件夹路径。
        output_file (str): 保存结果的 txt 文件路径。
        num_classes (int): 分割任务的类别数。
    """
    predictions = load_png_files(prediction_folder)
    labels = load_png_files(label_folder)

    # 确保文件夹中的文件名一致
    assert set(predictions.keys()) == set(labels.keys()), "预测结果和标签文件夹的文件名不一致！"

    overall_metrics = {
        "Dice": [],
        "IoU": [],
        "Recall": [],
        "Precision": [],
        "Hausdorff": []
    }

    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics per File:\n")

        for filename in predictions.keys():
            # if filename != '71.png':
            pred = binarize_image(predictions[filename])  # 二值化预测结果
            label = binarize_image(labels[filename])  # 二值化标签

            # 调整标签大小以匹配预测结果
            if pred.shape != label.shape:
                label = resize_to_match(label, pred.shape)

            metrics = calculate_metrics(pred, label, num_classes)

            # 写入每个文件的结果
            f.write(f"\nFile: {filename}\n")
            for key, value in metrics["Average"].items():
                f.write(f"{key}: {value:.4f}\n")

            if metrics["Hausdorff"][0] <= 224:
                for key in overall_metrics.keys():
                    overall_metrics[key].append(metrics["Average"][key])
            else:
                print(filename)

        # 计算总体平均值
        f.write("\nOverall Metrics:\n")
        for key in overall_metrics.keys():
            overall_metrics[key] = np.mean(overall_metrics[key])
            f.write(f"{key}: {overall_metrics[key]:.4f}\n")
            print(f"{overall_metrics[key]:.4f}")


    print(f"结果已保存至: {output_file}")

def evaluate_folder_255(prediction_folder, label_folder, output_file, num_classes=2):
    """
    评估两个文件夹中的 PNG 文件，并计算定量指标，同时将结果保存到指定位置。
    Args:
        prediction_folder (str): 存储预测结果的文件夹路径。
        label_folder (str): 存储标签的文件夹路径。
        output_file (str): 保存结果的 txt 文件路径。
        num_classes (int): 分割任务的类别数。
    """
    predictions = load_jpg_files_255(prediction_folder)
    labels = load_png_files(label_folder)

    # 确保文件夹中的文件名一致
    assert set(predictions.keys()) == set(labels.keys()), "预测结果和标签文件夹的文件名不一致！"

    overall_metrics = {
        "Dice": [],
        "IoU": [],
        "Recall": [],
        "Precision": [],
        "Hausdorff": []
    }

    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics per File:\n")

        for filename in predictions.keys():
            # if filename != '199.png':
            pred = binarize_image(predictions[filename])  # 二值化预测结果
            label = binarize_image(labels[filename])  # 二值化标签

            # 调整标签大小以匹配预测结果
            if pred.shape != label.shape:
                label = resize_to_match(label, pred.shape)

            metrics = calculate_metrics(pred, label, num_classes)

            # 写入每个文件的结果
            f.write(f"\nFile: {filename}\n")
            for key, value in metrics["Average"].items():
                f.write(f"{key}: {value:.4f}\n")

            if metrics["Hausdorff"][0] <= 224:
            # print(metrics["Hausdorff"])
                for key in overall_metrics.keys():
                    overall_metrics[key].append(metrics["Average"][key])
            else:
                print(filename)

        # 计算总体平均值
        f.write("\nOverall Metrics:\n")
        for key in overall_metrics.keys():
            overall_metrics[key] = np.mean(overall_metrics[key])
            f.write(f"{key}: {overall_metrics[key]:.4f}\n")
            print(f"{overall_metrics[key]:.4f}")


    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    # 预测结果文件夹路径
    prediction_folder = "/home/shuoxing/data/TransUNet/Results-Final/CVC-300/epoch_199"  # 替换为预测结果的文件夹路径

    # 标签文件夹路径
    label_folder = "/home/shuoxing/data/TransUNet/data/CVC-300/test/masks"  # 替换为标签的文件夹路径

    # 保存结果的文件路径
    output_file = "/home/shuoxing/data/TransUNet/Results-Final/results_CVC-300.txt"  # 替换为保存结果的路径

    # 设置类别数（包括背景）
    num_classes = 2

    # 计算并保存结果
    evaluate_folder(prediction_folder, label_folder, output_file, num_classes=num_classes)
