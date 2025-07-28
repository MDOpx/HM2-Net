import os
import logging
import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.data_ours import CVC_TestDataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import cv2
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 仅使用 GPU 0

# ==============================
# 🔹 解析命令行参数
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/home/shuoxing/data/TransUNet-original/TransUNet/dataset/h5/CVC-300',
                    help='Root directory for test volume data')
parser.add_argument('--model_dir', type=str,
                    default='/home/shuoxing/data/TransUNet-original/TransUNet/trained_model/UN-KAT-MFC/TU_CVC224_R50-ViT-B_16_skip3_vitpatch16_lr0.01_s1234',
                    help='Directory containing model checkpoint')
parser.add_argument('--dataset', type=str, default='CVC', help='Dataset name')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
parser.add_argument('--list_dir', type=str,
                    default='/home/shuoxing/data/TransUNet/lists/list_CVC-300',
                    help='Directory containing test file list')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
parser.add_argument('--img_size', type=int, default=224, help='Input patch size for the network')
parser.add_argument('--test_save_dir', type=str,
                    default='/home/shuoxing/data/TransUNet-original/TransUNet/Results/UN-KAT-MFC/CVC-300',
                    help='Directory to save main predictions')
parser.add_argument('--boundary_uncertainty_save_dir', type=str,
                    default='/home/shuoxing/data/TransUNet-original/TransUNet/Results/UN-KAT-MFC/CVC-300-Boundary-Uncertainty',
                    help='Directory to save boundary and uncertainty maps')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='Select ViT model')
parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patches size')
args = parser.parse_args()


# ==============================
# 🔹 推理函数
# ==============================
def inference(args, model, test_save_path, boundary_uncertainty_save_path):
    """ 对测试集进行推理，并分别保存主预测结果、边界图和不确定性信息 """

    db_test = CVC_TestDataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model.eval()

    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(boundary_uncertainty_save_path, exist_ok=True)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch['image'].cuda()
        case_name = sampled_batch['case_name'][0]

        if image.ndim != 4:
            raise ValueError(f"❌ 输入图像维度错误: {image.shape}")

        # ✅ 禁用梯度计算，提高推理效率
        with torch.no_grad():
            logits, boundary_map, mu, sigma = model(image)

        prediction = torch.argmax(logits, dim=1).cpu().numpy()

        # ✅ 处理 `boundary_map`
        boundary_map = boundary_map.squeeze(0).detach().cpu().numpy()
        if boundary_map.shape[0] >= 2:
            boundary_map = boundary_map[1]
        elif boundary_map.shape[0] == 1:
            boundary_map = boundary_map.squeeze(0)
        else:
            print(f"⚠️ Warning: Unexpected boundary_map shape: {boundary_map.shape}, using first channel.")
            boundary_map = boundary_map[0]

        # ✅ 处理 `uncertainty_map`
        uncertainty_map = mu.detach().cpu().numpy().squeeze()

        # ✅ 处理 `NaN` 和 `inf`
        boundary_map = np.nan_to_num(boundary_map)
        uncertainty_map = np.nan_to_num(uncertainty_map)

        # ✅ 保存
        save_path = os.path.join(test_save_path, f"{case_name}.png")
        boundary_path = os.path.join(boundary_uncertainty_save_path, f"{case_name}_boundary.png")
        uncertainty_path = os.path.join(boundary_uncertainty_save_path, f"{case_name}_uncertainty.png")

        cv2.imwrite(save_path, (prediction[0] * 255).astype(np.uint8))
        cv2.imwrite(boundary_path, (boundary_map * 255).astype(np.uint8))
        cv2.imwrite(uncertainty_path, (uncertainty_map * 255).astype(np.uint8))

        print(f"✅ 预测结果已保存: {save_path}")
        print(f"✅ 目标边界图已保存: {boundary_path}")
        print(f"✅ 不确定性图已保存: {uncertainty_path}")


# ==============================
# 🔹 运行测试
# ==============================
if __name__ == "__main__":
    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # ✅ 直接加载 `best_model.pth`
    model_path = os.path.join(args.model_dir, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ best_model.pth 文件未找到: {model_path}")

    print(f"🔄 正在加载模型 {model_path}...")

    # 加载 ViT 配置
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if 'R50' in args.vit_name:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)

    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # ✅ 加载 `best_model.pth`
    net.load_state_dict(torch.load(model_path))

    # ✅ 结果存放目录
    test_save_path = os.path.join(args.test_save_dir, "best_model_results")
    boundary_uncertainty_save_path = os.path.join(args.boundary_uncertainty_save_dir, "best_model_results")

    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(boundary_uncertainty_save_path, exist_ok=True)

    logging.info(f"Testing with best model: {model_path}")
    logging.info(f"Saving predictions to: {test_save_path}")
    logging.info(f"Saving boundary and uncertainty maps to: {boundary_uncertainty_save_path}")

    # ✅ 运行推理
    inference(args, net, test_save_path, boundary_uncertainty_save_path)

    print(f"✅ 推理完成，结果保存在 {test_save_path} 和 {boundary_uncertainty_save_path} ！")
