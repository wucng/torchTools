from .nms_pytorch import nms,nms2

from torch import nn
import torch
from torch.nn import functional as F
import random
import numpy as np
from .focalLoss import smooth_l1_loss_jit,giou_loss_jit,\
    sigmoid_focal_loss_jit,softmax_focal_loss_jit
import math
from math import sqrt

class SSDLoss(nn.Module):
    def __init__(self,cfg,device="cpu"):
        super(SSDLoss,self).__init__()
        self.device = device

        self.num_anchors = cfg["network"]["RPN"]["num_boxes"]
        self.num_classes = cfg["network"]["RPN"]["num_classes"]
        self.threshold_conf = cfg["work"]["loss"]["threshold_conf"]
        self.threshold_cls = cfg["work"]["loss"]["threshold_cls"]
        self.conf_thres = cfg["work"]["loss"]["conf_thres"]
        self.nms_thres = cfg["work"]["loss"]["nms_thres"]
        self.filter_labels = cfg["work"]["train"]["filter_labels"]
        self.useFocal = cfg["work"]["loss"]["useFocal"]
        self.alpha = cfg["work"]["loss"]["alpha"]
        self.gamma = cfg["work"]["loss"]["gamma"]
        self.strides = cfg["network"]["backbone"]["strides"]
        self.method = cfg["work"]["train"]["method"]

        self.resize = cfg["work"]["train"]["resize"]
        self.min_dim = cfg["network"]["prioriBox"]["min_dim"]
        scale = 1.0 * min(self.resize) / self.min_dim
        self.min_sizes = [int(c * scale) for c in cfg["network"]["prioriBox"]["min_sizes"]]
        self.max_sizes = [int(c * scale) for c in cfg["network"]["prioriBox"]["max_sizes"]]
        self.aspect_ratios = cfg["network"]["prioriBox"]["aspect_ratios"]
        self.variance = cfg["network"]["prioriBox"]["variance"]
        self.clip = cfg["network"]["prioriBox"]["clip"]
        self.thred_iou = cfg["network"]["prioriBox"]["thred_iou"]

        self.priorBox = self.get_prior_box() # [cx,cy,w,h]

        self.neg_pos_ratio = 3 # 正负样本比例1:3

    def get_prior_box(self):
        priors = []
        h, w = self.resize
        for idx, stride in enumerate(self.strides):
            fh, fw = h // stride, w // stride
            for i in range(fh):
                for j in range(fw):
                    # unit center x,y
                    cx = (j + 0.5) / fw
                    cy = (i + 0.5) / fh

                    # small sized square box
                    size_min = self.min_sizes[idx]
                    size_max = self.max_sizes[idx]
                    size = size_min
                    bh, bw = 1.0 * size / h, 1.0 * size / w
                    priors.append([cx, cy, bw, bh])

                    # big sized square box
                    size = sqrt(size_min * size_max)
                    bh, bw = 1.0 * size / h, 1.0 * size / w
                    priors.append([cx, cy, bw, bh])

                    # change h/w ratio of the small sized box
                    size = size_min
                    bh, bw = 1.0 * size / h, 1.0 * size / w
                    for ratio in self.aspect_ratios[idx]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, bw * ratio, 1.0 * bh / ratio])
                        priors.append([cx, cy, 1.0 * bw / ratio, bh * ratio])

        priors = torch.tensor(priors, device=self.device)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors

    def forward(self, preds, targets):
        if "boxes" not in targets[0]:
            results = self.predict(preds, targets)
            results = [self.apply_nms(result) for result in results]
            return results
        else:
            return self.compute_loss(preds, targets)

    def compute_loss(self, preds_list, targets_origin):
        last_result = self.normalize(targets_origin)
        pred_cls, pred_box = preds_list
        losses = {
            "loss_box": 0,
            "loss_clf": 0  # ,
            # "loss_iou":0
        }
        bs = len(targets_origin)
        for i in range(bs):
            confidence = pred_cls[i]
            predicted_locations = pred_box[i]
            gt_locations, labels = last_result[i]
            with torch.no_grad():
                loss = -F.log_softmax(confidence, dim=1)[:, 0]  # 计算背景置信度loss，后续需按loss从大到小排序，选择更新的mask
                mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

            confidence = confidence[mask, :]

            target_cls = F.one_hot(labels[mask],self.num_classes+1).to(self.device).float()
            classification_loss = sigmoid_focal_loss_jit(confidence, target_cls, self.alpha, self.gamma,
                                                         reduction='sum')

            # 定位loss 只更新正样本的
            pos_mask = labels > 0
            predicted_locations = predicted_locations[pos_mask, :]
            gt_locations = gt_locations[pos_mask, :]
            smooth_l1_loss = smooth_l1_loss_jit(predicted_locations, gt_locations, 2e-5, reduction='sum')

            losses["loss_box"] += smooth_l1_loss * 5.
            losses["loss_clf"] += classification_loss
        return losses

    def normalize(self, targets):
        last_result = []
        for target in targets:
            h, w = target["resize"]
            # gt_boxes = target["boxes"]  # (x1,y1,x2,y2)
            gt_labels = target["labels"]+1 # 背景为0

            # 缩减到图像上
            gt_boxes = target["boxes"]/torch.as_tensor([w,h,w,h],dtype=torch.float32, device=self.device).unsqueeze(0)

            # 与先验框计算IOU
            priorBoxXYXY = xywh2x1y1x2y2(self.priorBox)
            ious = box_iou(priorBoxXYXY, gt_boxes) # [N,M]
            # 有len(priorBoxXYXY)个先验框 按IOU分成前景与背景
            # size: num_priors
            best_target_per_prior, best_target_per_prior_index = ious.max(1)
            # 每个gt-box 对应一个priors-box(按IOU值最大分配)
            best_prior_per_target, best_prior_per_target_index = ious.max(0)
            for target_index,best_piror in enumerate(best_prior_per_target_index):
                best_target_per_prior_index[best_piror] = target_index
                best_target_per_prior[best_piror] = 1.0 # 确保每个gt_box对应得先验框为正样本

            labels = gt_labels[best_target_per_prior_index]
            boxes = gt_boxes[best_target_per_prior_index]
            labels[best_target_per_prior<self.thred_iou] = 0 # 小于阈值的设置0 即为背景

            # x1y1x2y2 --> xywh
            boxes = x1y1x2y22xywh(boxes)

            locations = convert_boxes_to_locations(boxes, self.priorBox,self.variance[0],self.variance[1])  # [8732,4]

            last_result.append([locations, labels])
        return last_result

    def predict(self, preds_list, targets_origin):
        """
        :param preds_list:
                   #（2个特征为例,batch=2）
                   preds_list=[(2,28,28,12),(2,14,14,12)]
        :param targets_origin:
                  [{"resize":(h,w),"origin_size":(h,w)},{"resize":(h,w),"origin_size":(h,w)}]
        :return:
        """
        pred_clses, pred_boxes = preds_list
        pred_clses = torch.sigmoid(pred_clses)
        result = []

        for i, (pred_cls, pred_box) in enumerate(zip(pred_clses, pred_boxes)):
            pred_box = self.reverse_normalize(pred_box)
            # scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
            scores, labels = pred_cls.max(dim=1)
            keep = (scores > self.threshold_cls) * (labels > 0)  # labels>0 跳过背景
            labels = labels - 1  # label从0 开始与yolo对应起来 排除掉背景
            if len(keep) == 0:
                pred_box = torch.zeros([1, 4], dtype=pred_box.dtype, device=pred_box.device)
                scores = torch.zeros([1, ], dtype=pred_box.dtype, device=pred_box.device)
                labels = torch.zeros([1, ], dtype=torch.long, device=pred_box.device)

            else:
                pred_box, scores, labels = pred_box[keep], scores[keep], labels[keep]


            result.append({"boxes": pred_box, "scores": scores, "labels": labels})
            result[i].update(targets_origin[i])

        return result

    def reverse_normalize(self, boxes):
        # [x0,y0,w,h]-->normalize 0~1--->[x1,y1,x2,y2]
        boxes = convert_locations_to_boxes(boxes, self.priorBox,self.variance[0],self.variance[1])
        boxes = xywh2x1y1x2y2(boxes)
        # 恢复到输入图像尺寸上
        resize = self.resize
        boxes = boxes * torch.as_tensor([resize[1], resize[0], resize[1], resize[0]],
                                        device=self.device,dtype=torch.float32).unsqueeze(0)

        # 裁剪到图像内
        boxes[..., [0, 2]] = torch.clamp(boxes[..., [0, 2]], 0, resize[1])
        boxes[..., [1, 3]] = torch.clamp(boxes[..., [1, 3]], 0, resize[0])

        return boxes

    def apply_nms(self, prediction):
        # for idx,prediction in enumerate(detections):
        # 1.先按scores过滤分数低的,过滤掉分数小于conf_thres
        ms = prediction["scores"] > self.conf_thres
        if torch.sum(ms) == 0:
            return None
        else:
            last_scores = []
            last_labels = []
            last_boxes = []

            # 2.类别一样的按nms过滤，如果Iou大于nms_thres,保留分数最大的,否则都保留
            # 按阈值过滤
            scores = prediction["scores"][ms]
            labels = prediction["labels"][ms]
            boxes = prediction["boxes"][ms]
            unique_labels = labels.unique()
            for c in unique_labels:
                if c in self.filter_labels: continue

                # Get the detections with the particular class
                temp = labels == c
                _scores = scores[temp]
                _labels = labels[temp]
                _boxes = boxes[temp]
                if len(_labels) > 1:
                    # Sort the detections by maximum objectness confidence
                    # _, conf_sort_index = torch.sort(_scores, descending=True)
                    # _scores=_scores[conf_sort_index]
                    # _boxes=_boxes[conf_sort_index]

                    # """
                    # keep=py_cpu_nms(_boxes.cpu().numpy(),_scores.cpu().numpy(),self.nms_thres)
                    keep = nms2(_boxes, _scores, self.nms_thres)
                    # keep = batched_nms(_boxes, _scores, _labels, self.nms_thres)
                    last_scores.extend(_scores[keep])
                    last_labels.extend(_labels[keep])
                    last_boxes.extend(_boxes[keep])

                else:
                    last_scores.extend(_scores)
                    last_labels.extend(_labels)
                    last_boxes.extend(_boxes)

            if len(last_labels) == 0:
                return None

            # resize 到原图上
            h_ori, w_ori = prediction["original_size"]
            h_re, w_re = prediction["resize"]
            h_ori = h_ori.float()
            w_ori = w_ori.float()

            # to pad图上
            if h_ori > w_ori:
                h_scale = h_ori / h_re
                w_scale = h_ori / w_re
                # 去除pad部分
                diff = h_ori - w_ori
                for i in range(len(last_boxes)):
                    last_boxes[i][[0, 2]] *= w_scale
                    last_boxes[i][[1, 3]] *= h_scale

                    last_boxes[i][0] -= diff // 2
                    last_boxes[i][2] -= diff - diff // 2

            else:
                h_scale = w_ori / h_re
                w_scale = w_ori / w_re
                diff = w_ori - h_ori
                for i in range(len(last_boxes)):
                    last_boxes[i][[0, 2]] *= w_scale
                    last_boxes[i][[1, 3]] *= h_scale

                    last_boxes[i][1] -= diff // 2
                    last_boxes[i][3] -= diff - diff // 2

            return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes}


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0 # 对应正样本
    num_pos = pos_mask.sum()
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf # 设置正样本的loss为无穷小，根据loss从到小排序选择负样本
    _, indexes = loss.sort(dim=0, descending=True)
    _, orders = indexes.sort(dim=0)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask

def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance=0.1, size_variance=0.2):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)

def convert_locations_to_boxes(locations, priors, center_variance=0.1,
                               size_variance=0.2):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x0, y0, x1, y1) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x0, y0, x1, y1) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def xywh2x1y1x2y2(boxes):
    """
    :param boxes: [...,4]
    :return:
    """
    x1y1=boxes[...,:2]-boxes[...,2:]/2
    x2y2=boxes[...,:2]+boxes[...,2:]/2

    return torch.cat((x1y1,x2y2),-1)

def x1y1x2y22xywh(boxes):
    """
    :param boxes: [...,4]
    :return:
    """
    xy=(boxes[...,:2]+boxes[...,2:])/2
    wh=boxes[...,2:]-boxes[...,:2]

    return torch.cat((xy,wh),-1)