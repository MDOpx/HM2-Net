import os
import logging
import argparse
import torch
from torchinfo import summary
from thop import profile
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.data_ours import CVC_TestDataset
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import cv2
import sys
from networks.uctnet_2D import uncertainty_map
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 仅使用 GPU 1

# CVC-300
# CVC-ClinicDB
# CVC-ColonDB
# ETIS-LaribPolypDB
# Kvasir
# BUSI
# UDIAT

def print_model_metrics(model, input_size=(1, 3, 352, 352)):
    # 创建虚拟输入
    dummy_input = torch.randn(*input_size).cuda()
    # model(dummy_input,'test')
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=[dummy_input,'test'])
    print(f"模型参数量: {params / 1e6:.2f}M")
    print(f"模型FLOPs: {flops / 1e9:.2f}G")

def cal_all(dataset_name):

    # dataset_name = 'CVC-300'
    dir_name = 'test'

    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str,
                        default='/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/h5',
                        help='Root directory for test volume data')
    parser.add_argument('--model_dir', type=str,
                        default='/home/shuoxing/data/TransUNet/model-finals-MFC/BUSI-352/TU_' + dataset_name + '352_pretrain_R50-ViT-B_16_skip3_epo1_bs1_352',
                        help='Directory containing model checkpoints')
    parser.add_argument('--dataset', type=str,
                        default=dataset_name, help='Dataset name')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='Number of output classes')
    parser.add_argument('--list_dir', type=str,
                        default='/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/lists',
                        help='Directory containing test file list')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--img_size', type=int, default=352, help='Input patch size for the network')
    parser.add_argument('--test_save_dir', type=str, default='/home/shuoxing/data/TransUNet/Results-Final-MFC/' + dataset_name + '/' + dir_name + '/epoch_200_att_onlymamba', help='Directory to save predictions')
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

            output, x, DS_T, UMap = model(image,casename = case_name)
            prediction = torch.argmax(output, dim=1).cpu().numpy()

            print('outputshape:',output.shape)

            _ = uncertainty_map(output.clone().detach(), num_classes=args.num_classes, casename=case_name)


            # save_path = os.path.join(test_save_path, f"{case_name}.png")
            # cv2.imwrite(save_path, (prediction[0] * 255).astype(np.uint8))
            # print(f"Saved prediction: {save_path}")

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

    # 指定最优模型的路径
    best_model_file = "best_model.pth"
    model_path = os.path.join(args.model_dir, best_model_file)
    print(f"🔄 正在加载最优模型 {model_path}...")

    # 加载 ViT 配置
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (args.img_size // args.vit_patches_size, args.img_size // args.vit_patches_size)
    # print(args.img_size)
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # 加载模型权重
    if not os.path.exists(model_path):
        print(f"⚠️ 模型文件未找到: {model_path}，跳过...")
    else:
        net.load_state_dict(torch.load(model_path))
        # 计算模型指标
        print_model_metrics(net, input_size=(1, 3, args.img_size, args.img_size))

        # 定义该模型的推理结果存放目录
        test_save_path = args.test_save_dir
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
        # inference(args, net, test_save_path)

    print(f"✅ 最优模型推理完成，结果保存在 {args.test_save_dir} 文件夹中！")


if __name__ == "__main__":
    cal_all('CVC-300')
    # cal_all('CVC-ClinicDB')
    # cal_all('CVC-ColonDB')
    # cal_all('ETIS-LaribPolypDB')
    # cal_all('Kvasir')
    # cal_all('BUSI')
    # cal_all('UDIAT')
    # cal_all('CTerwo')

# CVC-ClinicDB
# CVC-ColonDB
# ETIS-LaribPolypDB
# Kvasir
# BUSI
# UDIAT

'''
GRKAN+inceptionmamba
模型参数量: 106.95M
模型FLOPs: 78.32G
GRKAN
模型参数量: 128.22M
模型FLOPs: 88.58G
原
模型参数量: 128.22M
模型FLOPs: 88.58G

GRKAN+mamba
模型参数量: 106.95M
模型FLOPs: 78.32G
mamba
vit模型参数量: 5.32M
vit模型FLOPs: 2.57G
notmamba
vit模型参数量: 7.09M
vit模型FLOPs: 3.43G


'''


