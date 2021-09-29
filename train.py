import os.path

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from Datasets import YOLO_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import yaml
from Net import yolo1Net
from loss import YOLOV1_Loss
import warnings
from utils import *

warnings.filterwarnings("ignore")


def Data_augmented(mode="train") -> A.Compose:
    """
    数据增广的方法
    :return: 返回的是数据增强的手段，也就是albumentations中Compose的集合
    """
    img_transform = A.Compose
    if mode == "train":
        img_transform = A.Compose(
            [
                A.RGBShift(p=0.5),
                A.GaussNoise(p=0.5),
                A.Sharpen(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                A.CoarseDropout(p=0.5),
                A.Normalize(max_pixel_value=255),
                ToTensorV2(),
            ]
        )
    elif mode == "validation":
        img_transform = A.Compose(
            [
                A.Normalize(max_pixel_value=255),
                ToTensorV2(),
            ]
        )
    return img_transform


def generate_dataset(
        image_dir: str,
        label_dir: str,
        batch_size: int,
        mode: str) -> DataLoader:
    """
    这个方法主要用于输出训练集
    :param image_dir: 数据集的位置
    :param label_dir: 标签的位置
    :param batch_size: 每一个minibatch的大小
    :param mode: 这个代表的是输出数据集的格式，"train"为输出训练集格式，"val"

    :return: 返回的是一个Dataloader
    """
    # 数据增强手段
    img_transform = Data_augmented(mode)

    shuffle = True if mode == "train" else False
    dataloader = DataLoader(
        YOLO_dataset(
            image_dir,
            label_dir,
            mode,
            img_transform),
        shuffle=shuffle,
        pin_memory=True,  # 锁定一部分内存，这样可以加速训练
        batch_size=batch_size,
        num_workers=2,
    )

    return dataloader


def create_cfg(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./Voc/data.yaml',
                        help='please enter your dataset location(yaml)')
    parser.add_argument('--device', type=int, default=1, help='enter your GPU num 0,1,2...')
    parser.add_argument('--batch_size', type=int, default=2, help='number of the mini_batch 8,16,24,36,48...')
    parser.add_argument('--device_rank', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--model', type=str, default='./model/best.model', help='the path of the model')
    parser.add_argument('--iou', type=int, default=0.5, help='threshold value of iou')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def train(opt, model):
    loss_all = 0
    miou = 0
    with open(opt.data, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    train_dir = data['train']

    # 生成dataloader
    train_dataloader = generate_dataset(train_dir, train_dir.replace('images', 'labels'), opt.batch_size, 'train')
    # 生成模型
    model = model.to(torch.device(opt.device_rank))
    criterion = YOLOV1_Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.0005)
    lr_scadual = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    loop = tqdm(train_dataloader)
    for batch_index, (inputs, label) in enumerate(loop):
        inputs = inputs.to(torch.device(opt.device_rank))
        label = label.to(torch.device(opt.device_rank))
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # 评价模型
        # outputs[outputs[..., 10:] >= 0.5] = 1
        # outputs[outputs[..., 10:] < 0.5] = 0
        miou += MIOU(outputs, label).item()
        loss_all += loss.item()
        loop.set_postfix(loss=loss_all / (batch_index + 1), iou=miou / (batch_index + 1))
        lr_scadual.step()


def validation(opt, model):
    loss_all = 0
    miou = 0
    with open(opt.data, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    val_dir = data['val']

    # 生成验证集的dataloader
    val_dataloader = generate_dataset(val_dir, val_dir.replace('/images', '/labels'), opt.batch_size, 'validation')
    criterion = YOLOV1_Loss()
    model.eval()
    loop = tqdm(val_dataloader)
    with torch.no_grad():
        for batch_index, (inputs, label) in enumerate(loop):
            inputs = inputs.float().to(torch.device(opt.device_rank))
            label = label.to(torch.device(opt.device_rank))
            preds = model(inputs)
            # 经过softmax，将10:在0.5之上的元素设置为1
            # preds[preds[..., 10:] > 0.5] = 1
            # preds[preds[..., 10:] < 0.5] = 0
            loss = criterion(preds, label)
            miou += MIOU(preds, label)
            loss_all += loss
            loop.set_postfix(loss=loss_all / (batch_index + 1), iou=miou / (batch_index + 1))

    model.train()
    return miou


def main():
    best_score = 0.

    opt = create_cfg()
    if os.path.exists(opt.model):
        model = load_model(opt.model)

    else:
        model = yolo1Net()
    for i in range(opt.epoch):
        train(opt, model)
        miou = validation(opt, model)
        if miou > best_score:
            save_model(model, 'best_model')


if __name__ == '__main__':
    main()
