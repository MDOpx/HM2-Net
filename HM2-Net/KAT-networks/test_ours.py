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
import cv2  # Ensure OpenCV is installed
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 仅使用 GPU 1

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/home/shuoxing/data/TransUNet/dataset/h5/Kvasir',
                    help='Root directory for test volume data')
parser.add_argument('--model_dir', type=str,
                    default='/home/shuoxing/data/TransUNet/trained_model/KAT/TU_CVC224_pretrain_R50-ViT-B_16_skip3_epo200_bs8_224',
                    help='Directory containing model checkpoints')
parser.add_argument('--dataset', type=str,
                    default='CVC', help='Dataset name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='Number of output classes')
parser.add_argument('--list_dir', type=str,
                    default='/home/shuoxing/data/TransUNet/lists/list_Kvasir',
                    help='Directory containing test file list')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
parser.add_argument('--img_size', type=int, default=224, help='Input patch size for the network')
parser.add_argument('--test_save_dir', type=str, default='/home/shuoxing/data/TransUNet/Results/KAT/Kvasir', help='Directory to save predictions')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='Select ViT model')
parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patches size')
args = parser.parse_args()


def inference(args, model, test_save_path):
    """ 对测试集进行推理，并保存预测结果 """
    db_test = CVC_TestDataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model.eval()

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch['image'].cuda()
        case_name = sampled_batch['case_name'][0]

        if image.ndim != 4:
            raise ValueError(f"输入的图像张量维度不正确: {image.shape}, 期望维度为 [batch, channels, height, width].")

        output = model(image)
        prediction = torch.argmax(output, dim=1).cpu().numpy()

        save_path = os.path.join(test_save_path, f"{case_name}.png")
        cv2.imwrite(save_path, (prediction[0] * 255).astype(np.uint8))
        print(f"Saved prediction: {save_path}")


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

    # 获取所有模型文件（epoch_109.pth ~ epoch_199.pth）
    model_files = sorted(
        [f for f in os.listdir(args.model_dir) if f.startswith("epoch_") and f.endswith(".pth")],
        key=lambda x: int(x.split("_")[1].split(".")[0])  # 按照 epoch 号排序
    )

    for model_file in model_files:
        epoch_num = model_file.split("_")[1].split(".")[0]  # 提取 epoch 号
        model_path = os.path.join(args.model_dir, model_file)

        print(f"🔄 正在加载模型 {model_path}...")

        # 加载 ViT 配置
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)

        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        # 加载模型权重
        if not os.path.exists(model_path):
            print(f"⚠️ 模型文件未找到: {model_path}，跳过...")
            continue
        net.load_state_dict(torch.load(model_path))

        # 定义该模型的推理结果存放目录
        test_save_path = os.path.join(args.test_save_dir, f"KAT-{epoch_num}")
        os.makedirs(test_save_path, exist_ok=True)

        # 设置日志
        log_folder = os.path.join(test_save_path, "logs")
        os.makedirs(log_folder, exist_ok=True)
        log_file = os.path.join(log_folder, "test_log.txt")
        logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        logging.info(f"Testing with model: {model_path}")
        logging.info(f"Saving predictions to: {test_save_path}")

        # 运行推理
        inference(args, net, test_save_path)

    print(f"✅ 所有模型推理完成，结果保存在 {args.test_save_dir}/KAT-XXX 文件夹中！")
