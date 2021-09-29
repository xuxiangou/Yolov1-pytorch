import os
import random

import numpy as np
import cv2
import torch
from PIL import Image
from numpy import array
from torch import Tensor
from torch.utils.data import Dataset


bbox_num = 2
bbox_size = 5
class_num = 20
S = 7


def convert_normal_format(image, label, input_size=448) -> array:
    """
    将数据转为448×448的格式 （yolov1默认是448）
    （在这里默认所输入的图像都是小于448的......否则一定会存在信息损失）

    :param image: 输入读取的RGB图像矩阵
    :param input_size: 这个在yolov1中默认设置是448，可以根据图像实际大小进行调整
    :param label: 在这里标签的输入是多维（二维矩阵），并且是归一化的格式，因此我们处理也使用归一化进行处理即可
    :return:经过处理的image和标签
    """

    h, w, c = image.shape

    if h > w:
        pad_width = h - w
        image = np.pad(image, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
    elif w > h:
        pad_height = w - h
        image = np.pad(image, ((0, pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # 感觉算法中都直接用了resize，这样不会导致信息的丢失？
    # cv2.resize代表是对图像的缩放而不是简单的裁剪！！！
    # 未来的一个优化方向...对于一些大图来说，强行往小的去收缩，可能导致部分小物体检测效果差
    if image.shape[0] != 448 or image.shape[0] != 448:
        image = cv2.resize(image, (input_size, input_size))

    if len(label) != 0:
        # 在哪个方向进行扩展，就对哪个方向的坐标进行扩大然后再归一化
        if h > w:
            label[..., 1] = label[..., 1] * w / h
            label[..., 3] = label[..., 3] * w / h
        elif w > h:
            label[..., 2] = label[..., 2] * h / w
            label[..., 4] = label[..., 4] * h / w

    return image, label


def convert_boxLabel(bbox: array,
                     grid_size=1. / S,
                     offset=0.05) -> Tensor:
    target = torch.zeros(7, 7, int(bbox_num * bbox_size + class_num), dtype=torch.float32)
    # 有两个检测框，这两个的标签值设置为一样
    for i in range(len(bbox) // 5):
        gridx = int(bbox[i * 5 + 1] // grid_size)
        gridy = int(bbox[i * 5 + 2] // grid_size)

        gridpx = bbox[i * 5 + 1] / grid_size - gridx
        gridpy = bbox[i * 5 + 2] / grid_size - gridy

        target[gridy, gridx, 0:5] = torch.FloatTensor([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        target[gridy, gridx, 5:10] = torch.FloatTensor([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        target[gridy, gridx, int(10 + bbox[i * 5])] = 1
    return target


class YOLO_dataset(Dataset):
    """
    这里输入的label的格式为：(tensor) --> (n,5)  其中5为c,x,y,w,h
    这里的x,y,w,h都是归一化的结果，x,y是矩形框中心的归一化坐标，w,h是归一化后的矩形框的大小
    """

    def __init__(self,
                 image_file: str,
                 label_file: str,
                 mode: str,
                 image_transform=None):
        super(YOLO_dataset, self).__init__()
        self.image_file = image_file
        self.label_file = label_file
        self.mode = mode
        self.image_list = os.listdir(self.image_file)
        self.image_transform = image_transform

    def __getitem__(self, index) -> [Tensor]:
        # images = cv2.imread(self.image_file + '/' + self.image_list[index])
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        image = np.array(Image.open(self.image_file + '/' + self.image_list[index]).convert('RGB'))
        # cv2.imshow('images',images)
        # cv2.waitKey(0)
        label = np.loadtxt(
            self.label_file + '/' + os.path.splitext(self.image_list[index])[0] + '.txt')

        image, label = convert_normal_format(image, label)

        # 将target转换为(7,7,2*B+class_num)的格式
        target = convert_boxLabel(label.flatten())

        # 这里进行一个模式的判断
        if self.mode == 'train':
            if self.image_transform is not None:
                args = self.image_transform(image=image)
                image = args['image']
        elif self.mode == 'validation':
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
        else:
            raise RuntimeError(f'模式输入错误，应该使用validation或者是train格式，但是你使用了{self.mode}')

        return image, target

    def __len__(self) -> int:
        return len(self.image_list)
