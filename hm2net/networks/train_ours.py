import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer_ours import trainer_synapse  # 确保 trainer_synapse 支持你的训练逻辑
from datasets.data_ours import CVC_dataset, RandomGenerator  # 替换为你的数据加载器和增强器
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 仅使用 GPU 1

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/shuoxing/data/TransUNet-original/TransUNet/dataset/npz/train', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='CVC', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='/home/shuoxing/data/TransUNet-original/TransUNet/lists/lists_CVC', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum number of iterations to train')
parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch size per GPU')
parser.add_argument('--n_gpu', type=int, default=1, help='total number of GPUs')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether to use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='learning rate for the segmentation network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of the network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='number of skip connections to use')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one ViT model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='ViT patches size, default is 16')
args = parser.parse_args()

if __name__ == "__main__":
    # 确保训练的可重复性
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'CVC': {
            'root_path': '/home/shuoxing/data/TransUNet-original/TransUNet/dataset/npz/train',
            'list_dir': '/home/shuoxing/data/TransUNet-original/TransUNet/lists/lists_CVC',
            'num_classes': 2,
        },
    }

    # 加载对应数据集的配置信息
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    # 设置实验名和模型保存路径
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "/home/shuoxing/data/TransUNet-original/TransUNet/trained_model/UN-KAT/{}".format(args.exp)
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # 创建保存路径
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # 配置 ViT 模型
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip

    # 设置预训练权重路径
    config_vit.pretrained_path = "/home/shuoxing/data/TransUNet-original/TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"

    # 检查权重文件是否存在
    if not os.path.exists(config_vit.pretrained_path):
        raise FileNotFoundError(f"预训练权重文件未找到: {config_vit.pretrained_path}")

    # 设置 ViT patch grid
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))

    # 初始化模型
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    # 加载预训练权重
    net.load_from(weights=np.load(config_vit.pretrained_path))

    # 初始化训练和验证数据集
    train_dataset = CVC_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size])
    )
    val_dataset = CVC_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="val",
        transform=RandomGenerator(output_size=[args.img_size, args.img_size])
    )

    # 定义训练函数映射
    trainer = {
        'Synapse': trainer_synapse,
        'CVC': trainer_synapse,
    }

    # 开始训练
    trainer[dataset_name](args, net, snapshot_path, train_dataset, val_dataset)

