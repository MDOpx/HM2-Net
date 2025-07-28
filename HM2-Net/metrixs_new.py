import os
import numpy as np
from PIL import Image
from skimage.transform import resize
from scipy.ndimage import sobel


def load_png_files(folder_path):
    """
    加载 PNG 图像，并返回文件名到二值化图像数组的映射。
    """
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图
            image = (image > 0).astype(np.uint8)  # 二值化
            data[filename] = image
    return data


def resize_to_match(image, target_shape):
    """
    调整图像的尺寸以匹配目标形状。
    """
    return resize(image, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)


def dice_score(pred, gt):
    """
    计算 Dice 系数
    """
    intersection = np.sum(pred * gt)
    dice = (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)
    return dice


def iou_score(pred, gt):
    """
    计算 IoU (交并比)
    """
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-6)
    return iou


def structure_measure(pred, gt):
    """
    计算 Structure Measure (Sα)
    结合区域相似性 (Region Similarity) 和 边缘相似性 (Edge Similarity)
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    # 计算区域相似性（Dice Score）
    intersection = np.sum(pred * gt)
    region_similarity = (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)

    # 计算边缘相似性（使用 Sobel 进行边缘检测）
    pred_edges = np.abs(sobel(pred))
    gt_edges = np.abs(sobel(gt))

    edge_intersection = np.sum(pred_edges * gt_edges)
    edge_union = np.sum(pred_edges) + np.sum(gt_edges)
    edge_similarity = edge_intersection / (edge_union + 1e-6)

    # 组合区域相似性和边缘相似性
    S_alpha = 0.95 * region_similarity + 0.05 * edge_similarity

    return S_alpha


def e_measure(pred, gt):
    """
    计算 Mean E-measure (Eφ)
    """
    smooth = 1e-6
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    E_phi = np.mean((pred - gt) ** 2)
    return 1 - np.sqrt(E_phi + smooth)


def weighted_f_measure(pred, gt, beta=0.3):
    """
    计算 Weighted F-measure (Fβ^ω)
    """
    TP = np.sum(pred * gt)
    FP = np.sum(pred * (1 - gt))
    FN = np.sum((1 - pred) * gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    weight = 2 * (precision * recall) / (precision + recall + 1e-6)

    F_beta_w = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall + 1e-6)
    return weight * F_beta_w


def mean_absolute_error(pred, gt):
    """
    计算 Mean Absolute Error (MAE)
    确保 MAE 计算在 0-1 之间
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mae = np.sum(np.abs(pred - gt)) / (pred.shape[0] * pred.shape[1])
    return mae


def calculate_metrics(predictions, labels):
    """
    计算论文中的六种评估指标。
    """
    metrics = {
        "Dice": [],
        "IoU": [],
        "S_alpha": [],
        "E_phi": [],
        "F_beta_w": [],
        "MAE": []
    }

    for filename in predictions.keys():
        pred = predictions[filename]
        gt = labels[filename]

        # 确保形状匹配
        if pred.shape != gt.shape:
            gt = resize_to_match(gt, pred.shape)

        # 计算所有指标
        metrics["Dice"].append(dice_score(pred, gt))
        metrics["IoU"].append(iou_score(pred, gt))
        metrics["S_alpha"].append(structure_measure(pred, gt))
        metrics["E_phi"].append(e_measure(pred, gt))
        metrics["F_beta_w"].append(weighted_f_measure(pred, gt))
        metrics["MAE"].append(mean_absolute_error(pred, gt))

    # 计算平均值
    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    metrics["Average"] = avg_metrics

    return metrics


def evaluate_folder(prediction_folder, label_folder, output_file):
    """
    评估预测结果文件夹与标签文件夹，计算六种评估指标，并保存结果。
    """
    predictions = load_png_files(prediction_folder)
    labels = load_png_files(label_folder)

    assert set(predictions.keys()) == set(labels.keys()), "预测结果和标签文件夹的文件名不一致！"

    overall_metrics = {
        "Dice": [],
        "IoU": [],
        "S_alpha": [],
        "E_phi": [],
        "F_beta_w": [],
        "MAE": []
    }

    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics per File:\n")

        for filename in predictions.keys():
            metrics = calculate_metrics({filename: predictions[filename]}, {filename: labels[filename]})

            f.write(f"\nFile: {filename}\n")
            for key, value in metrics["Average"].items():
                f.write(f"{key}: {value:.4f}\n")

            for key in overall_metrics.keys():
                overall_metrics[key].append(metrics["Average"][key])

        # 计算总体平均值
        f.write("\nOverall Metrics:\n")
        for key in overall_metrics.keys():
            overall_metrics[key] = np.mean(overall_metrics[key])
            f.write(f"{key}: {overall_metrics[key]:.4f}\n")

    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    prediction_folder = "/home/shuoxing/data/TransUNet/Results-Final-MFC/CVC-ColonDB/epoch_199"
    label_folder = "/home/shuoxing/data/TransUNet/data/CVC-ColonDB/test/masks"
    output_file = "/home/shuoxing/data/TransUNet/Results-Final-MFC/results_CVC-ColonDB.txt"

    evaluate_folder(prediction_folder, label_folder, output_file)
