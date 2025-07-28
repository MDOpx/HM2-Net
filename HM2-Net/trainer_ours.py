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
from networks.deep_supervision import MultipleOutputLoss2
from networks.dice_loss import DC_and_CE_loss
from networks.to_torch import maybe_to_torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def test_model(model, testloader, ce_loss, dice_loss):
    """
    Evaluate the model on the test set and compute Dice score.

    Args:
        model: The trained model.
        testloader: DataLoader for the test dataset.
        ce_loss: CrossEntropyLoss function.
        dice_loss: DiceLoss function.

    Returns:
        avg_test_loss: Average loss on the test set.
        avg_test_dice: Average Dice coefficient on the test set.
    """
    model.eval()
    test_loss, test_ce, test_dice_score = 0, 0, 0
    test_batches = len(testloader)

    test_bar = tqdm(enumerate(testloader), total=test_batches, desc="Testing", ncols=100, leave=True)

    with torch.no_grad():
        for _, sampled_batch in test_bar:
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            # 获取模型的主要输出
            outputs, *_ = model(image_batch,casename = sampled_batch['case_name'][0])

            # 计算损失
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            test_loss += loss.item()
            test_ce += loss_ce.item()

            # ✅ **修正 Dice 计算方式（适用于二分类）**
            prob = F.softmax(outputs, dim=1)[:, 1]  # 获取前景概率
            pred = (prob > 0.5).float()  # 进行二值化
            label_batch = label_batch.float().squeeze(1)  # 确保维度匹配

            # 计算 Dice Score
            intersection = torch.sum(pred * label_batch)
            dice_score = (2.0 * intersection) / (torch.sum(pred) + torch.sum(label_batch) + 1e-5)
            test_dice_score += dice_score.item()

    # 计算最终平均值
    avg_test_loss = test_loss / test_batches
    avg_test_dice = test_dice_score / test_batches

    # ✅ 记录日志
    logging.info(f"Test - Loss: {avg_test_loss:.4f}, Dice: {avg_test_dice:.4f}")

    # ✅ 终端打印最终结果
    print(f"\nTest - Loss: {avg_test_loss:.4f}, Dice: {avg_test_dice:.4f}")

    return avg_test_loss, avg_test_dice



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
    # print(batch_size)
    # 初始化 DataLoader
    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 多 GPU 支持
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    # 定义损失函数和优化器
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    loss_utc = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    uct_loss = MultipleOutputLoss2(loss_utc)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

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

            # 前向传播
            outputs = model(image_batch)

            # 计算损失
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice


            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 学习率调整
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            # 记录日志
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            logging.info(f'iteration {iter_num}: loss: {loss.item():.4f}, loss_ce: {loss_ce.item():.4f}')




        # 保存模型
        save_interval = 10  # 每 50 个 epoch 保存一次
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


def trainer_synapse_utc(args, model, snapshot_path, train_dataset, test_dataset):
    """
    Trainer with Dice evaluation on the test dataset.
    """
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"),
                        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Total training batches: {len(trainloader)}")
    print(f"Total test batches: {len(testloader)}")
    sys.stdout.flush()

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # loss_utc = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    # uct_loss = MultipleOutputLoss2(loss_utc)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    best_dice = 0.0

    logging.info(f"Total Epochs: {max_epoch}, Training Batches per Epoch: {len(trainloader)}")

    for epoch_num in range(max_epoch):
        model.train()
        total_loss, total_ce, total_dice, total_uct, total_dice_score = 0, 0, 0, 0, 0
        num_batches = len(trainloader)

        logging.info(f"Epoch {epoch_num+1}/{max_epoch} started.")
        print(f"\n=== Epoch {epoch_num+1}/{max_epoch} started ===")
        sys.stdout.flush()

        train_bar = tqdm(enumerate(trainloader), total=num_batches, desc=f"Epoch {epoch_num+1}/{max_epoch}", ncols=100)

        for i_batch, sampled_batch in train_bar:
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            outputs, x, DS_T, UMap = model(image_batch,casename = sampled_batch['case_name'][0])

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            label_batch = label_batch.unsqueeze(1)
            target_size = image_batch.shape[-2:]

            # outputs_loss, target_loss, DS_T_loss, UMap_loss = [], [], [], []
            # for item in x:
            #     outputs_loss.append(F.adaptive_avg_pool2d(item, target_size))
            #     target_loss.append(torch.cat([1 - label_batch, label_batch], dim=1))
            #
            # for item in DS_T:
            #     DS_T_loss.append(F.interpolate(item, size=target_size, mode='bilinear', align_corners=False))
            #
            # for item in UMap:
            #     out_down = F.interpolate(item.unsqueeze(1).float(), size=target_size, mode='nearest').long()
            #     UMap_loss.append(out_down)
            #
            # loss_tar = uct_loss(outputs_loss, target_loss, DS_T_loss, UMap_loss)
            # loss += 0.1 * loss_tar

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_dice += loss_dice.item()
            # total_uct += loss_tar.item()

            pred = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            dice_score = 2.0 * torch.sum(pred * label_batch.squeeze(1)) / (torch.sum(pred) + torch.sum(label_batch.squeeze(1)) + 1e-5)
            total_dice_score += dice_score.item()

        scheduler.step()

        avg_loss = total_loss / num_batches
        avg_ce = total_ce / num_batches
        avg_dice = total_dice / num_batches
        # avg_uct = total_uct / num_batches
        avg_dice_score = total_dice_score / num_batches

        # ✅ 仅在 `epoch` 结束后打印 Loss 和 Dice
        print(f"=== Epoch {epoch_num+1}/{max_epoch} Finished ===")
        # print(f"Train Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Dice Loss: {avg_dice:.4f}, UCT Loss: {avg_uct:.4f}, Train Dice: {avg_dice_score:.4f}")
        print(
            f"Train Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Dice Loss: {avg_dice:.4f}, Train Dice: {avg_dice_score:.4f}")
        sys.stdout.flush()
        # logging.info(
            # f"Epoch {epoch_num + 1}/{max_epoch} - Train Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Dice Loss: {avg_dice:.4f}, UCT Loss: {avg_uct:.4f}, Train Dice: {avg_dice_score:.4f}")
        logging.info(f"Epoch {epoch_num+1}/{max_epoch} - Train Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Dice Loss: {avg_dice:.4f}, Train Dice: {avg_dice_score:.4f}")

        # ✅ 只在 `epoch` 结束后进行测试集评估
        print(f"Starting evaluation on test dataset after Epoch {epoch_num+1}")
        sys.stdout.flush()
        avg_test_loss, avg_test_dice = test_model(model, testloader, ce_loss, dice_loss)
        print(f"Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}, Best Dice:{best_dice:.4f}")
        sys.stdout.flush()

        logging.info(f"Finished evaluation: Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}, Best Dice:{best_dice:.4f}")

        if avg_test_dice > best_dice:
            best_dice = avg_test_dice
            best_model_path = os.path.join(snapshot_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated! Dice: {best_dice:.4f}")
            sys.stdout.flush()
            logging.info(f"Best model updated! Dice: {best_dice:.4f}")

    print("\n=== Training Completed! ===")
    print(f"Best Dice Score Achieved: {best_dice:.4f}")
    logging.info(f"Training Finished! Best Dice Score: {best_dice:.4f}")
    final_model_path = os.path.join(snapshot_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"final model updated! Dice: {best_dice:.4f}")

    return "Training Finished!"



