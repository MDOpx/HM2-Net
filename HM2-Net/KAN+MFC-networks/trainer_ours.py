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
    """
    if softmax:
        preds = torch.softmax(preds, dim=1)

    targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))
    union = torch.sum(preds, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
    dice = (2. * intersection + 1e-5) / (union + 1e-5)  # 避免除零
    return dice.mean().detach()


def trainer_synapse(args, model, snapshot_path, train_dataset, val_dataset):
    """
    训练器（Trainer）用于 Synapse 或自定义数据集。
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

    trainloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    iter_num = 0
    max_epoch = args.max_epochs
    best_dice = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        epoch_loss = 0.0
        total_train_dice = 0.0
        num_train_samples = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            optimizer.zero_grad()
            outputs = model(image_batch)

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            total_loss = 0.5 * loss_ce + 0.5 * loss_dice
            epoch_loss += total_loss.item()

            train_dice = dice_score(outputs, label_batch, num_classes=num_classes, softmax=True)
            total_train_dice += train_dice.item()
            num_train_samples += 1

            total_loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('info/total_loss', total_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/train_dice', train_dice, iter_num)

        avg_train_loss = epoch_loss / len(trainloader)
        avg_train_dice = total_train_dice / num_train_samples

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('info/lr', current_lr, epoch_num)

        # **打印训练集上的损失和 Dice**
        logging.info(f"Epoch {epoch_num}: Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")

        # **从第 100 个 epoch 开始进行验证**
        if epoch_num >= 100:
            model.eval()
            total_val_dice = 0.0
            num_val_samples = 0
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in valloader:
                    image_val, label_val = val_batch['image'], val_batch['label']
                    image_val, label_val = image_val.cuda(), label_val.cuda()

                    val_outputs = model(image_val)

                    loss_ce_val = ce_loss(val_outputs, label_val.long())
                    loss_dice_val = dice_loss(val_outputs, label_val, softmax=True)
                    total_val_loss = 0.5 * loss_ce_val + 0.5 * loss_dice_val
                    val_loss += total_val_loss.item()

                    dice_score_value = dice_score(val_outputs, label_val, num_classes=num_classes, softmax=True)
                    total_val_dice += dice_score_value.item()
                    num_val_samples += 1

            avg_val_loss = val_loss / num_val_samples
            avg_val_dice = total_val_dice / num_val_samples

            writer.add_scalar('info/val_dice', avg_val_dice, epoch_num)
            writer.add_scalar('info/val_loss', avg_val_loss, epoch_num)

            # **打印验证集上的损失和 Dice**
            logging.info(f"Epoch {epoch_num}: Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

            # **保存最佳模型**
            if avg_val_dice > best_dice:
                best_dice = avg_val_dice
                best_model_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Best model saved with Dice Score: {best_dice:.4f}")

    # **训练结束后，在验证集上进行最终测试**
    model.eval()
    total_val_dice = 0.0
    num_val_samples = 0
    val_loss = 0.0
    with torch.no_grad():
        for val_batch in valloader:
            image_val, label_val = val_batch['image'], val_batch['label']
            image_val, label_val = image_val.cuda(), label_val.cuda()

            val_outputs = model(image_val)
            dice_score_value = dice_score(val_outputs, label_val, num_classes=num_classes, softmax=True)
            total_val_dice += dice_score_value.item()
            num_val_samples += 1

    avg_final_val_dice = total_val_dice / num_val_samples
    logging.info(f"Final Validation Dice Score: {avg_final_val_dice:.4f}")

    writer.close()
    return "Training Finished!"
