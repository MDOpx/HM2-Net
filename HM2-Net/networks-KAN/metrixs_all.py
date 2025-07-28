import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from skimage.transform import resize

# 设定预测结果的固定尺寸
PREDICTION_SIZE = (224, 224)


def load_png_files(folder_path, resize_shape=None):
    """ 加载 PNG 图像，并返回字典 {文件名: 图像数组}，可选 resize 处理 """
    data = {}
    if not os.path.exists(folder_path):
        print(f"⚠️  文件夹不存在: {folder_path}")
        return data
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图

            # 统一 resize labels
            if resize_shape:
                image = resize(image, resize_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

            data[filename] = image
    return data


def binarize_image(image):
    """ 二值化图像 """
    return (image > 0).astype(np.uint8)


def hausdorff_distance(A, B, max_dist=100):
    """ 计算 Hausdorff 95 距离，处理空集情况，避免 inf"""
    A_points = np.argwhere(A)
    B_points = np.argwhere(B)

    if len(A_points) == 0 or len(B_points) == 0:
        return max_dist  # 如果预测或标签为空，则返回固定的大值

    d_matrix = cdist(A_points, B_points)
    return np.percentile(d_matrix, 95)  # 计算 95% Hausdorff 距离


def calculate_metrics(predictions, labels, num_classes=2):
    """ 计算分割指标（Dice, IoU, Recall, Precision, Hausdorff）"""
    metrics = {"Dice": [], "IoU": [], "Recall": [], "Precision": [], "Hausdorff": []}

    for cls in range(1, num_classes):  # 忽略背景（class 0）
        cls_preds = (predictions == cls).astype(int)
        cls_labels = (labels == cls).astype(int)

        intersection = np.sum(cls_preds * cls_labels)
        dice = (2. * intersection) / (np.sum(cls_preds) + np.sum(cls_labels) + 1e-6)
        metrics["Dice"].append(dice)

        union = np.sum(cls_preds) + np.sum(cls_labels) - intersection
        iou = intersection / (union + 1e-6)
        metrics["IoU"].append(iou)

        recall = intersection / (np.sum(cls_labels) + 1e-6)
        precision = intersection / (np.sum(cls_preds) + np.sum(cls_labels) + 1e-6)
        metrics["Recall"].append(recall)
        metrics["Precision"].append(precision)

        # 计算 Hausdorff 95 距离
        hausdorff = hausdorff_distance(cls_preds, cls_labels)

        if hausdorff == 100:  # 之前的 inf 现在变成 100
            print(f"⚠️ Hausdorff 计算异常: 预测或标签为空，返回 {hausdorff}")

        metrics["Hausdorff"].append(hausdorff)

    return {key: np.mean(value) for key, value in metrics.items()}  # 计算均值


def evaluate_best_model(predictions_root, label_folder, output_file, best_model_folder, num_classes=2):
    """ 评估指定的最佳模型的推理结果 """

    prediction_folder = os.path.join(predictions_root, best_model_folder)

    if not os.path.exists(prediction_folder):
        print(f"⚠️ 预测结果文件夹不存在: {prediction_folder}，跳过...")
        return

    predictions = load_png_files(prediction_folder)  # 预测结果不需要 resize
    labels = load_png_files(label_folder, resize_shape=PREDICTION_SIZE)  # 统一 resize labels

    common_files = set(predictions.keys()) & set(labels.keys())
    if not common_files:
        print(f"⚠️  {best_model_folder} 无匹配的预测和标签文件，跳过...")
        return

    print(f"📊 正在计算 {best_model_folder} 的指标...")

    overall_metrics = {"Dice": [], "IoU": [], "Recall": [], "Precision": [], "Hausdorff": []}

    for filename in common_files:
        pred = binarize_image(predictions[filename])
        label = binarize_image(labels[filename])

        metrics = calculate_metrics(pred, label, num_classes)

        for key in overall_metrics.keys():
            overall_metrics[key].append(metrics[key])

    # 计算该模型的总体均值
    avg_metrics = {key: np.mean(value) for key, value in overall_metrics.items()}

    # 写入日志
    with open(output_file, 'w') as f:
        f.write(f"\nModel: {best_model_folder}\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"✅ {best_model_folder} 计算完成！")
    print(f"📌 评估完成，结果已保存至 {output_file}")


if __name__ == "__main__":
    # 你的多个数据集的路径，格式为 (预测结果路径, 标签路径, 结果文件路径)
    datasets = [
        ("/home/shuoxing/data/TransUNet/Results/KAT-1/CVC-300",
         "/home/shuoxing/data/TransUNet/dataset/TestDataset/CVC-300/masks",
         "/home/shuoxing/data/TransUNet/Results/KAT-1/TxT/all_results_CVC-300.txt"),

        ("/home/shuoxing/data/TransUNet/Results/KAT-1/CVC-ClinicDB",
         "/home/shuoxing/data/TransUNet/dataset/TestDataset/CVC-ClinicDB/masks",
         "/home/shuoxing/data/TransUNet/Results/KAT-1/TxT/all_results_ClinicDB.txt"),

        ("/home/shuoxing/data/TransUNet/Results/KAT-1/CVC-ColonDB",
         "/home/shuoxing/data/TransUNet/dataset/TestDataset/CVC-ColonDB/masks",
         "/home/shuoxing/data/TransUNet/Results/KAT-1/TxT/all_results_CVC-ColonDB.txt"),

        ("/home/shuoxing/data/TransUNet/Results/KAT-1/ETIS",
         "/home/shuoxing/data/TransUNet/dataset/TestDataset/ETIS-LaribPolypDB/masks",
         "/home/shuoxing/data/TransUNet/Results/KAT-1/TxT/all_results_ETIS.txt"),

        ("/home/shuoxing/data/TransUNet/Results/KAT-1/Kvasir",
         "/home/shuoxing/data/TransUNet/dataset/TestDataset/Kvasir/masks",
         "/home/shuoxing/data/TransUNet/Results/KAT-1/TxT/all_results_Kvasir.txt"),
    ]

    best_model_folder = "KAT-best"  # 指定最佳模型文件夹

    for pred_path, label_path, output_file in datasets:
        print(f"\n🔍 开始评估最佳模型: {pred_path}")
        evaluate_best_model(pred_path, label_path, output_file, best_model_folder, num_classes=2)
