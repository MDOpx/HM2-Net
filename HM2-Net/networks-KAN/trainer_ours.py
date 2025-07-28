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


def dice_score(preds, targets, num_classes, softmax=True):
    """
    计算 Dice Score（评估用，不用于训练）。
    :param preds: 预测值 (batch, num_classes, H, W)
    :param targets: 真实标签 (batch, H, W)
    :param num_classes: 类别数
    :param softmax: 是否对预测结果应用 softmax
    :return: Dice Score（越接近1越好）
    """
    if softmax:
        preds = torch.softmax(preds, dim=1)  # 应用 softmax 以得到概率

    targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))  # 计算交集
    union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))  # 计算并集

    dice = (2. * intersection + 1e-5) / (union + 1e-5)  # 避免除零
    return dice.mean().detach()  # 返回 Dice Score（非 Loss）


def trainer_synapse(args, model, snapshot_path, train_dataset, val_dataset):
    """
    训练器（Trainer）用于 Synapse 或自定义数据集。

    Args:
        args: 训练参数与配置。
        model: 需要训练的模型。
        snapshot_path: 模型与日志保存路径。
        train_dataset: 训练数据集。
        val_dataset: 验证数据集。
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
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # 余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations.")
    best_dice = 0.0  # 记录最佳 Dice Score

    # 训练循环
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            # 加载训练数据
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            optimizer.zero_grad()

            # 前向传播
            outputs = model(image_batch)

            # 计算损失（Dice Loss + CrossEntropy Loss）
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)

            # 组合总损失
            total_loss = 0.5 * loss_ce + 0.5 * loss_dice
            epoch_loss += total_loss.item()

            # 反向传播
            total_loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                f'Iteration {iter_num}: loss: {total_loss.item():.4f}, loss_ce: {loss_ce.item():.4f}, '
                f'loss_dice: {loss_dice.item():.4f}'
            )

        # 调整学习率（余弦退火）
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('info/lr', current_lr, iter_num)

        # 进行验证
        model.eval()
        total_dice = 0.0
        num_samples = 0
        with torch.no_grad():
            for val_batch in valloader:
                image_val, label_val = val_batch['image'], val_batch['label']
                image_val, label_val = image_val.cuda(), label_val.cuda()

                val_outputs = model(image_val)
                dice_score_value = dice_score(val_outputs, label_val, num_classes=args.num_classes, softmax=True)

                total_dice += dice_score_value.item()
                num_samples += 1

        avg_dice = total_dice / num_samples  # 计算平均 Dice Score
        writer.add_scalar('info/val_dice', avg_dice, epoch_num)
        logging.info(f"Epoch {epoch_num}: Avg Dice Score: {avg_dice:.4f}")

        # 保存最佳模型
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved with Dice Score: {best_dice:.4f}")

        # 每 10 个 epoch 保存一次
        if (epoch_num + 1) % 10 == 0:
            checkpoint_path = os.path.join(snapshot_path, f'epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved model checkpoint at {checkpoint_path}")

        # 保存最终模型
        if epoch_num == max_epoch - 1:
            final_model_path = os.path.join(snapshot_path, 'final_model.pth')
            torch.save(model.state_dict(), final_model_path)
            logging.info(f"Final model saved at {final_model_path}")

    writer.close()
    return "Training Finished!"
