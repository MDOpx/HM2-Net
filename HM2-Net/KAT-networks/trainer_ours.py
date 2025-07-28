import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from utils import DiceLoss  # 确保 DiceLoss 正确导入


def trainer_synapse(args, model, snapshot_path, train_dataset, val_dataset):
    """
    Trainer for Synapse or custom datasets.

    Args:
        args: Training arguments and configurations.
        model: The model to be trained.
        snapshot_path: Path to save the model and logs.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
    """
    # 配置日志记录
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # 初始化 DataLoader
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 多 GPU 支持
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # 定义损失函数和优化器
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    cos_sim = nn.CosineSimilarity(dim=1)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations.")
    best_performance = 0.0

    # 主训练循环
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            # 加载训练数据
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            optimizer.zero_grad()

            # 前向传播
            outputs = model(image_batch)

            # 计算分割损失（Dice Loss + CrossEntropy Loss）
            loss_ce = ce_loss(outputs, label_batch.long())  # 处理类别索引格式
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            # 组合总损失
            total_loss = 0.5 * loss_ce + 0.5 * loss_dice

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 学习率调整
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # 记录日志
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)


            logging.info(
                f'iteration {iter_num}: loss: {total_loss.item():.4f}, loss_ce: {loss_ce.item():.4f}, '
                f'loss_dice: {loss_dice.item():.4f}'
            )

        # 保存模型
        save_interval = 10  # 每 5 个 epoch 保存一次
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved model to {save_mode_path}")

        # 最终保存
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved model to {save_mode_path}")
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
