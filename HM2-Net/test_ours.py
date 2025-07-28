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
from networks.uctnet_2D import uncertainty_map
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 仅使用 GPU 1

# CVC-300
# CVC-ClinicDB
# CVC-ColonDB
# ETIS-LaribPolypDB
# Kvasir
# BUSI
# UDIAT

dataset_name = 'CVC-300'

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/home/shuoxing/data/TransUNet/data/' + dataset_name + '/test/h5',
                    help='Root directory for test volume data')
parser.add_argument('--model_dir', type=str,
                    default='/home/shuoxing/data/TransUNet/model-finals-MFC/BUSI-352/TU_' + dataset_name + '352_pretrain_R50-ViT-B_16_skip3_epo1_bs2_352',
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
parser.add_argument('--test_save_dir', type=str, default='/home/shuoxing/data/TransUNet/Results-Final-MFC/' + dataset_name + '/epoch_500', help='Directory to save predictions')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='Select ViT model')
parser.add_argument('--deterministic', type=int, default=1, help='Deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patches size')
args = parser.parse_args()

gradients = []
activations = []

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output


def inference(args, model, test_save_path):
    """ 对测试集进行推理，并保存预测结果 """
    db_test = CVC_TestDataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # print(model)
    # # *************************************************************************************
    # global activations
    # global gradients
    # # target_layer = model.decoder.blocks[2].vit_blocks.recover_patch_embedding[1]
    # target_layer = model.decoder.blocks[3].conv2[2]
    # target_layer.register_forward_hook(forward_hook)
    # target_layer.register_backward_hook(backward_hook)
    # # *************************************************************************************
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch['image'].cuda()
        case_name = sampled_batch['case_name'][0]

        if image.ndim != 4:
            raise ValueError(f"输入的图像张量维度不正确: {image.shape}, 期望维度为 [batch, channels, height, width].")

        # output = model(image)
        # print(image.shape)
        output, x, DS_T, UMap = model(image,casename = case_name)
        prediction = torch.argmax(output, dim=1).cpu().numpy()

        # ✅ 调用不确定性图保存
        # _ = uncertainty_map(output.clone().detach(), num_classes=args.num_classes, casename=case_name)

        save_path = os.path.join(test_save_path, f"{case_name}.png")
        cv2.imwrite(save_path, (prediction[0] * 255).astype(np.uint8))
        print(f"Saved prediction: {save_path}")

    # #     # *************************************************************************************
    #     model.zero_grad()
    #     # print(output.requires_grad)
    #     target_class = torch.argmax(output, dim=1)
    #     output[:, target_class].sum().backward()
    #     gradients = gradients[0].squeeze(0)
    #     activations = activations[0].squeeze(0)
    #     alpha = gradients.mean(dim=(1, 2), keepdim=True)
    #     cam = (alpha * activations).sum(dim=0).detach().cpu().numpy()
    #     # cam = (activations).sum(dim=0).detach().cpu().numpy()
    #     cam = np.maximum(cam, 0)
    #     cam = (cam - cam.min()) / (cam.max() - cam.min())
    #     cam = cv2.resize(cam, (224, 224))  # 调整大小匹配输入图像
    #     cam[cam>1]=1
    #     heatmap = cv2.applyColorMap(np.uint8(255 - 255 * cam), cv2.COLORMAP_JET)
    #     save_path = os.path.join('/home/shuoxing/data/TransUNet/Results-Final-MFC/attn_map/', f"{case_name}.png")
    #     cv2.imwrite(save_path, heatmap)
    #     # plt.subplot(1, 3, 3)
    #     # plt.imshow(heatmap, cmap='jet')
    #     # plt.title("Grad-CAM on Fusion Feature Map")
    #     # plt.axis("off")
    #     # plt.show()
    # # # *************************************************************************************


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
        inference(args, net, test_save_path)

    print(f"✅ 最优模型推理完成，结果保存在 {args.test_save_dir} 文件夹中！")
