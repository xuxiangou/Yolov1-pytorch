import torch
from torch import nn, Tensor

S = 7.


def save_model_dict(model, name) -> None:
    torch.save(model.state_dict(), f'./model/{name}.pth')


def save_model(model, name: str) -> None:
    torch.save(model, f'./model/{name}.pth')


def load_model(path: str):
    model = torch.load(path)
    return model


def precise(y_pr: Tensor, y_gt: Tensor, threshold_iou):
    precision = 0.
    num = 1
    for batch_size in range(y_pr.shape[0]):
        for i in range(y_pr.shape[1]):
            for j in range(y_pr.shape[2]):
                if y_gt[batch_size, i, j, 4] == 1:
                    t = 0 if y_pr[batch_size, i, j, 4] > y_pr[batch_size, i, j, 9] else 1
                    iou = IOU(y_pr[batch_size, i, j, t:t + 4], y_gt[batch_size, i, j, t:t + 4])
                    if iou > threshold_iou:
                        positive_pred_index = y_gt[batch_size, i, j, 10:] == 1
                        positive_correct = \
                            (y_gt[batch_size, i, j, 10:][positive_pred_index] == y_pr[batch_size, i, j, 10:][
                                positive_pred_index]).shape[0]
                        # 1e-8是国际惯例，防止分母为0
                        precision += positive_correct / (
                                y_gt[..., i, j, 10:][positive_pred_index].shape[0] * num + 1e-8)
                    num += 1
    return precision


def recall(y_pr: Tensor, y_gt: Tensor, threshold_iou):
    recall = 0.
    num = 1
    for batch_size in range(y_pr.shape[0]):
        for i in range(y_pr.shape[1]):
            for j in range(y_pr.shape[2]):
                if y_gt[batch_size, i, j, 4] == 1:
                    t = 0 if y_pr[batch_size, i, j, 4] > y_pr[batch_size, i, j, 9] else 1
                    iou = IOU(y_pr[batch_size, i, j, t:t + 4], y_gt[batch_size, i, j, t:t + 4])
                    if iou > threshold_iou:
                        positive_pred_index = y_gt[batch_size, i, j, 10:] == 1
                        positive_correct = \
                            (y_gt[batch_size, i, j, 10:][positive_pred_index] == y_pr[batch_size, i, j, 10:][
                                positive_pred_index]).shape[0]
                        # 1e-8是国际惯例，防止分母为0
                        negative_pred_index = y_gt[batch_size, i, j, 10:] != 1
                        negative_false = \
                            (y_gt[batch_size, i, j, 10:][negative_pred_index] != y_pr[batch_size, i, j, 10:][
                                negative_pred_index]).shape[0]
                        recall += positive_correct / (
                                (positive_correct + negative_false) * num + 1e-8)
                    num += 1
    return recall


def IOU(box1: Tensor, box2: Tensor, ):
    """
    这个函数是计算两个盒子之间的iou值
    计算方法：取得两个矩形方框的交集和并集之比
    :param box1:盒子1的参数，(x1c,y1c,w1,h1)
    :param box2:盒子2的参数  (x2c,y2c,w2,h2)
    :return: 返回的是一个张量
    """

    # 找到相交的矩形框，去除不相交的情况

    x1 = torch.min(box1[0], box2[0])  # box1xmax,box2xmax
    x2 = torch.max(box1[2], box2[2])
    y1 = torch.min(box1[1], box2[1])
    y2 = torch.max(box1[3], box2[3])

    intersection_w = x1 - x2
    intersection_h = y1 - y2

    if intersection_w <= 0 or intersection_h <= 0:
        return 0

    bbox_intersection = intersection_h * intersection_w
    bbox_union = (box1[0] - box1[2]) * (box1[1] - box1[3]) + (box2[0] - box2[2]) * (box2[1] - box2[3])
    return bbox_intersection / (bbox_union - bbox_intersection + 1e-8)


def MIOU(y_pr: Tensor, y_gt: Tensor, ):
    iou = 0
    num = 0
    for batch_size in range(y_pr.shape[0]):
        for i in range(y_pr.shape[1]):
            for j in range(y_pr.shape[2]):
                if y_gt[batch_size, i, j, 4] == 1:
                    t = 0 if IOU(y_gt[batch_size, i, j, 0:4], y_pr[batch_size, i, j, 0:4]) > IOU(
                        y_gt[batch_size, i, j, 5:9], y_pr[batch_size, i, j, 5:9]) else 1

                    bbox_pr_xxyy = torch.Tensor([
                        (y_pr[batch_size, i, j, t * 5] + j) / S + y_pr[batch_size, i, j, t * 5 + 2] / 2,
                        (y_pr[batch_size, i, j, t * 5 + 1] + i) / S + y_pr[batch_size, i, j, t * 5 + 3] / 2,
                        (y_pr[batch_size, i, j, t * 5] + j) / S - y_pr[batch_size, i, j, t * 5 + 2] / 2,
                        (y_pr[batch_size, i, j, t * 5 + 1] + i) / S - y_pr[batch_size, i, j, t * 5 + 3] / 2,
                    ])
                    bbox_gt_xxyy = torch.Tensor([
                        (y_gt[batch_size, i, j, t * 5] + j) / S + y_gt[batch_size, i, j, t * 5 + 2] / 2,
                        (y_gt[batch_size, i, j, t * 5 + 1] + i) / S + y_gt[batch_size, i, j, t * 5 + 3] / 2,
                        (y_gt[batch_size, i, j, t * 5] + j) / S - y_gt[batch_size, i, j, t * 5 + 2] / 2,
                        (y_gt[batch_size, i, j, t * 5 + 1] + i) / S - y_gt[batch_size, i, j, t * 5 + 3] / 2,
                    ])

                    iou += IOU(bbox_gt_xxyy, bbox_pr_xxyy)
                    num += 1
    return iou / (num + 1e-8)

# if __name__ == '__main__':
#     y_gt = torch.zeros([6, 7, 7, 30])
#     y_pr = torch.ones([6, 7, 7, 30])
#     y_gt[..., 2, 4, 4] = 1
#     y_gt[..., 2, 3, 4] = 1
#     print(MIOU(y_pr, y_gt))
