import os
import random

import numpy as np
import cv2
import torch
from numpy import array
from torch import Tensor

S = 7

# a = np.loadtxt('000_1ov3n5_0_jpeg.rf.a23f1c89491779996f4519858277a4e0.txt')
# print(a)


#
# def convert_boxLabel(bbox: array,
#                      grid_size=1. / S) -> array:
#     target = np.zeros([7, 7, 30], dtype=np.float32)
#     # 有两个检测框，这两个的标签值设置为一样
#     for i in range(len(bbox) // 5):
#         gridx = int(bbox[i * 5 + 1] // grid_size)
#         gridy = int(bbox[i * 5 + 2] // grid_size)
#
#         gridpx = bbox[i * 5 + 1] / grid_size - gridx
#         gridpy = bbox[i * 5 + 2] / grid_size - gridy
#
#         target[gridx, gridy, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
#         target[gridx, gridy, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
#         target[gridx, gridy, 10 + int(bbox[i * 5])] = 1
#
#     return target

# def IOU(box1: Tensor, box2: Tensor) -> Tensor:
#     """
#     这个函数是计算两个盒子之间的iou值
#     计算方法：取得两个矩形方框的交集和并集之比
#     :param box1:盒子1的参数，(x1c,y1c,w1,h1)
#     :param box2:盒子2的参数  (x2c,y2c,w2,h2)
#     :return: 返回的是一个张量
#     """
#
#     # box1的位置信息
#     box1_xmax = box1[..., 0] + box1[..., 2] / 2
#     box1_ymax = box1[..., 1] + box1[..., 3] / 2
#     box1_xmin = box1[..., 0] - box1[..., 2] / 2
#     box1_ymin = box1[..., 1] - box1[..., 3] / 2
#
#
#     # box2的位置信息
#     box2_xmax = box2[..., 0] + box2[..., 2] / 2
#     box2_ymax = box2[..., 1] + box2[..., 3] / 2
#     box2_xmin = box2[..., 0] - box2[..., 2] / 2
#     box2_ymin = box2[..., 1] - box2[..., 3] / 2
#
#
#     # 找到相交的矩形框，去除不相交的情况
#     intersection_w = torch.min(box1_xmax, box2_xmax) - torch.max(box1_xmin, box2_xmin)
#     intersection_h = torch.min(box1_ymax, box2_ymax) - torch.max(box1_ymin, box2_ymin)
#     intersection_w[intersection_w < 0] = 0
#     intersection_h[intersection_h < 0] = 0
#
#     bbox_intersection = intersection_w * intersection_h
#     bbox_union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3]
#     return (bbox_intersection ) / (bbox_union - bbox_intersection )


if __name__ == '__main__':
    # a = np.loadtxt('000_1ov3n5_0_jpeg.rf.a23f1c89491779996f4519858277a4e0.txt')
    # a = a.flatten()
    # target = convert_boxLabel(a)
    # index = (np.where(target[...,0]!=0))
    # print(target[index][...,0])
    # box1 = torch.tensor([[5, 4, 3, 2], [13, 14, 1, 1]], dtype=torch.float64)
    # box2 = torch.tensor([[4, 3, 2, 2], [19, 19, 2, 2]], dtype=torch.float64)
    # print(IOU(box1, box2))

    # images = cv2.imread('F:/python_coding/torch/yolo/yolov5/Mask_data/train/images/1553605632_9d5877d8_60_jpg.rf.827ad9a39f672aba0756c9ac68d25239.jpg')
    # cv2.imshow('images',images)
    # cv2.waitKey(0)
    target = torch.zeros([7, 7, 30], dtype=torch.float32)
    target[...,15] = 1
    index = target[...,10:] == 1
    pred = torch.zeros([7, 7, 30], dtype=torch.float32)
    pred[1,1,15] = 1

    print(pred[...,10:][index])

# random.seed(42)
# print(random.randint(1,10))
