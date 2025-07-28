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
from utils import DiceLoss


# === 新增代码：定义 KL 散度损失 ===
class UncertaintyLoss(nn.Module):
    """不确定性损失 (KL 散度)"""
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, mu, sigma):
        # 计算 KL 散度
        kl_loss = torch.mean(0.5 * (mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1))
        return kl_loss


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
        filename=snapshot_path + "/log.txt",
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
    uncertainty_loss = UncertaintyLoss()  # === 新增：不确定性损失 ===
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    # 主训练循环
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            # 加载训练数据
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # === 修改前向传播，增加不确定性输出 ===
            logits, boundary_map, mu, sigma = model(image_batch)

            # 计算 Dice + CrossEntropy Loss
            loss_ce = ce_loss(logits, label_batch.long())
            loss_dice = dice_loss(logits, label_batch, softmax=True)

            # === 计算边界损失 ===
            loss_boundary = dice_loss(boundary_map, label_batch, softmax=True)

            # === 计算不确定性损失（KL 散度） ===
            loss_uncertainty = uncertainty_loss(mu, sigma)

            # === 计算总损失 ===
            loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.3 * loss_boundary + 0.2 * loss_uncertainty

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 学习率调整
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # 记录日志
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_boundary', loss_boundary, iter_num)  # === 记录边界损失 ===
            writer.add_scalar('info/loss_uncertainty', loss_uncertainty, iter_num)  # === 记录不确定性损失 ===
            logging.info(f'iteration {iter_num}: loss: {loss.item():.4f}, loss_ce: {loss_ce.item():.4f}, '
                         f'loss_dice: {loss_dice.item():.4f}, loss_boundary: {loss_boundary.item():.4f}, '
                         f'loss_uncertainty: {loss_uncertainty.item():.4f}')

        # 保存模型
        save_interval = 10  # 每 10 个 epoch 保存一次
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
