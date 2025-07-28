import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize


def load_png_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图
            data[filename] = image
    return data


def binarize_image(image):
    return (image > 0).astype(np.uint8)


def resize_to_match(image, target_shape):
    resized_image = resize(image, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    return resized_image


def calculate_metrics(predictions, labels, num_classes=2):
    metrics = {
        "Dice": [], "IoU": [], "Recall": [], "Precision": [], "Hausdorff": [],
        "Accuracy": [], "NPV": [], "Jaccard": [], "FPR": []
    }
    for cls in range(1, num_classes):  # 从 1 开始计算，跳过背景类
        cls_preds = (predictions == cls).astype(int)
        cls_labels = (labels == cls).astype(int)
        TP = np.sum(cls_preds * cls_labels)
        FP = np.sum(cls_preds * (1 - cls_labels))
        FN = np.sum((1 - cls_preds) * cls_labels)
        TN = np.sum((1 - cls_preds) * (1 - cls_labels))

        dice = (2. * TP) / (2 * TP + FP + FN + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        npv = TN / (TN + FN + 1e-6)
        jaccard = iou
        fpr = FP / (FP + TN + 1e-6)

        metrics["Dice"].append(dice)
        metrics["IoU"].append(iou)
        metrics["Recall"].append(recall)
        metrics["Precision"].append(precision)
        metrics["Accuracy"].append(accuracy)
        metrics["NPV"].append(npv)
        metrics["Jaccard"].append(jaccard)
        metrics["FPR"].append(fpr)

        if cls_preds.sum() > 0 and cls_labels.sum() > 0:
            hausdorff = max(
                directed_hausdorff(cls_preds, cls_labels)[0],
                directed_hausdorff(cls_labels, cls_preds)[0]
            )
        else:
            hausdorff = np.inf
        metrics["Hausdorff"].append(hausdorff)

    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    metrics["Average"] = avg_metrics
    return metrics


def evaluate_folder(prediction_folder, label_folder, output_file, num_classes=2):
    predictions = load_png_files(prediction_folder)
    labels = load_png_files(label_folder)
    assert set(predictions.keys()) == set(labels.keys()), "预测结果和标签文件夹的文件名不一致！"

    overall_metrics = {key: [] for key in
                       ["Dice", "IoU", "Recall", "Precision", "Hausdorff", "Accuracy", "NPV", "Jaccard", "FPR"]}

    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics per File:\n")
        for filename in predictions.keys():
            pred = binarize_image(predictions[filename])
            label = binarize_image(labels[filename])

            if pred.shape != label.shape:
                label = resize_to_match(label, pred.shape)

            metrics = calculate_metrics(pred, label, num_classes)

            f.write(f"\nFile: {filename}\n")
            for key, value in metrics["Average"].items():
                f.write(f"{key}: {value:.4f}\n")

            for key in overall_metrics.keys():
                overall_metrics[key].append(metrics["Average"][key])

        f.write("\nOverall Metrics:\n")
        for key in overall_metrics.keys():
            overall_metrics[key] = np.mean(overall_metrics[key])
            f.write(f"{key}: {overall_metrics[key]:.4f}\n")

    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    prediction_folder = "/home/shuoxing/data/TransUNet/Results-Final-MFC/UDIAT/epoch_199"
    label_folder = "/home/shuoxing/data/TransUNet-original/TransUNet/data/UDIAT/testing/output"
    output_file = "/home/shuoxing/data/TransUNet/Results-Final-MFC/results_UDIAT.txt"
    num_classes = 2
    evaluate_folder(prediction_folder, label_folder, output_file, num_classes=num_classes)
import os
import numpy as np
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize


def load_png_files(folder_path):
    data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            image = np.array(Image.open(file_path).convert('L'))  # 转为灰度图
            data[filename] = image
    return data


def binarize_image(image):
    return (image > 0).astype(np.uint8)


def resize_to_match(image, target_shape):
    resized_image = resize(image, target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    return resized_image


def calculate_metrics(predictions, labels, num_classes=2):
    metrics = {
        "Dice": [], "IoU": [], "Recall": [], "Precision": [], "Hausdorff": [],
        "Accuracy": [], "NPV": [], "Jaccard": [], "FPR": []
    }
    for cls in range(1, num_classes):  # 从 1 开始计算，跳过背景类
        cls_preds = (predictions == cls).astype(int)
        cls_labels = (labels == cls).astype(int)
        TP = np.sum(cls_preds * cls_labels)
        FP = np.sum(cls_preds * (1 - cls_labels))
        FN = np.sum((1 - cls_preds) * cls_labels)
        TN = np.sum((1 - cls_preds) * (1 - cls_labels))

        dice = (2. * TP) / (2 * TP + FP + FN + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        npv = TN / (TN + FN + 1e-6)
        jaccard = iou
        fpr = FP / (FP + TN + 1e-6)

        metrics["Dice"].append(dice)
        metrics["IoU"].append(iou)
        metrics["Recall"].append(recall)
        metrics["Precision"].append(precision)
        metrics["Accuracy"].append(accuracy)
        metrics["NPV"].append(npv)
        metrics["Jaccard"].append(jaccard)
        metrics["FPR"].append(fpr)

        if cls_preds.sum() > 0 and cls_labels.sum() > 0:
            hausdorff = max(
                directed_hausdorff(cls_preds, cls_labels)[0],
                directed_hausdorff(cls_labels, cls_preds)[0]
            )
        else:
            hausdorff = np.inf
        metrics["Hausdorff"].append(hausdorff)

    avg_metrics = {key: np.mean(value) for key, value in metrics.items()}
    metrics["Average"] = avg_metrics
    return metrics


def evaluate_folder(prediction_folder, label_folder, output_file, num_classes=2):
    predictions = load_png_files(prediction_folder)
    labels = load_png_files(label_folder)
    assert set(predictions.keys()) == set(labels.keys()), "预测结果和标签文件夹的文件名不一致！"

    overall_metrics = {key: [] for key in
                       ["Dice", "IoU", "Recall", "Precision", "Hausdorff", "Accuracy", "NPV", "Jaccard", "FPR"]}

    with open(output_file, 'w') as f:
        f.write("Evaluation Metrics per File:\n")
        for filename in predictions.keys():
            pred = binarize_image(predictions[filename])
            label = binarize_image(labels[filename])

            if pred.shape != label.shape:
                label = resize_to_match(label, pred.shape)

            metrics = calculate_metrics(pred, label, num_classes)

            f.write(f"\nFile: {filename}\n")
            for key, value in metrics["Average"].items():
                f.write(f"{key}: {value:.4f}\n")

            for key in overall_metrics.keys():
                overall_metrics[key].append(metrics["Average"][key])

        f.write("\nOverall Metrics:\n")
        for key in overall_metrics.keys():
            overall_metrics[key] = np.mean(overall_metrics[key])
            f.write(f"{key}: {overall_metrics[key]:.4f}\n")

    print(f"结果已保存至: {output_file}")


if __name__ == "__main__":
    prediction_folder = "/home/shuoxing/data/TransUNet/Results-Final-MFC/BUSI/epoch_199"
    label_folder = "/home/shuoxing/data/TransUNet-original/TransUNet/data/BUSI/testing/output"
    output_file = "/home/shuoxing/data/TransUNet/Results-Final-MFC/results_BUSI.txt"
    num_classes = 2
    evaluate_folder(prediction_folder, label_folder, output_file, num_classes=num_classes)
