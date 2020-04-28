"""
# yolo ：num_classes 不包括背景
# SSD ：num_classes 包括背景（背景为0）

try:
    from .boxestool import batched_nms
except:
    from boxestool import batched_nms
"""
import sys,os
from .nms_pytorch import nms,nms2
from .focalLoss import smooth_l1_loss_jit,giou_loss_jit,\
    sigmoid_focal_loss_jit,softmax_focal_loss_jit

from torch import nn
import torch
from torch.nn import functional as F
import random
import numpy as np
from math import sqrt
import math


class SSDLoss(nn.Module):
    def __init__(self,device="cpu",num_anchors=6,
                 num_classes=20, # 不包括背景（+1包括背景）
                 threshold_conf=0.05,
                 threshold_cls=0.5,
                 conf_thres=0.8,
                 nms_thres=0.4,
                 filter_labels:list = [],
                 mulScale=False):
        super(SSDLoss, self).__init__()
        self.device = device
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.threshold_conf = threshold_conf
        self.threshold_cls = threshold_cls
        self.mulScale = mulScale
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.filter_labels = filter_labels
        self.neg_pos_ratio = 3

    def forward(self,preds,targets):
        if "boxes" not in targets[0]:
            # return self.predict(preds,targets)
            results = self.predict(preds,targets)
            results = [self.apply_nms(result) for result in results]
            return results
        else:
            return self.compute_loss(preds, targets,useFocal=True)

    def compute_loss(self,preds_list, targets_origin,useFocal=False,alpha=0.2,gamma=2):
        """
        :param preds:
                if mulScale: # 使用多尺度（2个特征为例,batch=2）
                    preds=[[(1,28,28,12),(1,14,14,12)],[(1,28,28,12),(1,14,14,12)]]
                else: #（2个特征为例,batch=2）
                   preds=[(2,28,28,12),(2,14,14,12)]
        :param targets:
                [{"boxes":(n,4),"labels":(n,)},{"boxes":(m,4),"labels":(m,)}]
        :return:
        """
        losses = {
            "loss_box": 0,
            "loss_clf": 0,
            "loss_iou":0
        }

        for jj in range(len(targets_origin)):
            target_origin = targets_origin[jj]
            if self.mulScale:
                pred_list = preds_list[jj]
            else:
                pred_list =[pred[jj].unsqueeze(0) for pred in preds_list]

            for i, preds in enumerate(pred_list):
                fh, fw = preds.shape[1:-1]
                # normalize
                gt_locations,labels = self.normalize((fh, fw), target_origin)

                if gt_locations is None:
                    smooth_l1_loss = 0*F.mse_loss(torch.rand([1,2],device=self.device).detach(),torch.rand(1,2,device=self.device).detach(),reduction="sum")
                    classification_loss = 0*F.mse_loss(torch.rand([1,2],device=self.device).detach(),torch.rand(1,2,device=self.device).detach(),reduction="sum")
                else:
                    preds = preds.contiguous().view(-1,5 + self.num_classes)
                    preds[..., :2] = torch.sigmoid(preds[..., :2])

                    confidence = preds[...,4:] # 包括背景（背景与类别放在一起做，yolo则是分开做）
                    predicted_locations = preds[...,:4]

                    with torch.no_grad():
                        # derived from cross_entropy=sum(log(p))
                        loss = -F.log_softmax(confidence, dim=1)[:, 0]  # 计算背景置信度loss，后续需按loss从大到小排序，选择更新的mask
                        mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

                    confidence = confidence[mask, :]
                    if useFocal:
                        classification_loss = softmax_focal_loss_jit(confidence, labels[mask], alpha, gamma,
                                                                     reduction='sum')
                    else:
                        classification_loss = F.cross_entropy(confidence, labels[mask], reduction='sum')


                    # 定位loss 只更新正样本的
                    pos_mask = labels > 0
                    predicted_locations = predicted_locations[pos_mask, :]
                    gt_locations = gt_locations[pos_mask, :]
                    # smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
                    smooth_l1_loss = smooth_l1_loss_jit(predicted_locations, gt_locations,2e-5, reduction='sum')

                    loss_iou = giou_loss_jit(xywh2x1y1x2y2(predicted_locations),xywh2x1y1x2y2(gt_locations),reduction='sum')


                losses["loss_box"] += smooth_l1_loss * 5.
                losses["loss_clf"] += classification_loss
                losses["loss_iou"] += loss_iou * 5.

        return losses

    def get_prior_box(self, featureShape, target,clip=True):
        priors = []
        fh,fw=featureShape
        h, w = target["resize"]
        stride = 1.0*h/fh
        for i in range(fh):
            for j in range(fw):
                # unit center x,y
                cx = (j+0.5)/fw
                cy = (i+0.5)/fh

                # small sized square box
                size_min = 30*stride/4.
                size_max = 60*stride/4.
                size = size_min
                bh,bw = 1.0*size/h,1.0*size/w
                priors.append([cx,cy,bw,bh])

                # big sized square box
                size = sqrt(size_min * size_max)
                bh,bw = 1.0*size/h,1.0*size/w
                priors.append([cx, cy, bw, bh])

                # change h/w ratio of the small sized box
                size = size_min
                bh, bw = 1.0*size / h, 1.0*size / w
                for ratio in [2,3]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, bw * ratio, 1.0*bh / ratio])
                    priors.append([cx, cy, 1.0*bw / ratio, bh * ratio])

        priors = torch.tensor(priors,device=self.device)
        if clip:
            priors.clamp_(max=1, min=0)
        return priors

    def normalize(self, featureShape, target):
        fh, fw = featureShape
        center_form_priors = self.get_prior_box((fh, fw), target)
        priors_x1y1x2y2 = xywh2x1y1x2y2(center_form_priors)

        gt_boxes = target["boxes"]  # (x1,y1,x2,y2)
        if len(gt_boxes)==0:return None,None
        if gt_boxes.dim()<2:gt_boxes = gt_boxes.unsqueeze(0)

        gt_labels = target["labels"]+1 # 背景为0
        input_size = target["resize"]
        # 缩放到输入图像
        gt_boxes = gt_boxes / torch.as_tensor(
            [input_size[1], input_size[0], input_size[1], input_size[0]],
            dtype=torch.float32, device=self.device).unsqueeze(0)


        boxes, labels = assign_priors(gt_boxes, gt_labels, priors_x1y1x2y2)

        # x1y1x2y2 --> xywh
        boxes = x1y1x2y22xywh(boxes)

        locations = convert_boxes_to_locations(boxes, center_form_priors)  # [8732,4]

        return locations,labels


    def predict(self, preds_list,targets_origin):
        """
        :param preds_list:
                   #（2个特征为例,batch=2）
                   preds_list=[(2,28,28,12),(2,14,14,12)]
        :param targets_origin:
                  [{"resize":(h,w),"origin_size":(h,w)},{"resize":(h,w),"origin_size":(h,w)}]
        :return:
        """
        result = []

        for idx, preds in enumerate(preds_list):
            bs, fh, fw = preds.shape[:-1]
            preds = preds.contiguous().view(bs,-1, 5 + self.num_classes)
            preds[...,:2] = torch.sigmoid(preds[...,:2])

            for i in range(bs):
                targets = targets_origin[i]
                new_preds = preds[i]
                new_preds[...,:4] = self.reverse_normalize((fh, fw), new_preds[...,:4], targets)

                pred_box = new_preds[:,:4]
                pred_cls = new_preds[:, 4:] # 包括背景
                scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
                # keep = torch.nonzero(scores > self.threshold_cls and labels>0) # labels>0 跳过背景
                keep = (scores > self.threshold_cls) * (labels>0) # labels>0 跳过背景
                labels = labels-1 # label从0 开始与yolo对应起来

                if len(keep)==0:
                    pred_box = torch.zeros([1, 4], dtype=pred_box.dtype, device=pred_box.device)
                    scores = torch.zeros([1, 1], dtype=pred_box.dtype, device=pred_box.device)
                    labels = torch.zeros([1, 1], dtype=pred_box.dtype, device=pred_box.device)
                else:
                    pred_box, scores, labels = pred_box[keep], scores[keep], labels[keep]

                if len(result) < bs:
                    result.append({"boxes": pred_box, "scores": scores, "labels": labels})
                    result[i].update(targets)
                else:
                    assert len(result) == bs, print("error")
                    result[i]["boxes"] = torch.cat((result[i]["boxes"], pred_box), 0)
                    result[i]["scores"] = torch.cat((result[i]["scores"], scores), 0)
                    result[i]["labels"] = torch.cat((result[i]["labels"], labels), 0)

        return result

    def reverse_normalize(self,featureShape,boxes, target):
        # [x0,y0,w,h]-->normalize 0~1--->[x1,y1,x2,y2]
        priors = self.get_prior_box(featureShape,target)
        boxes = convert_locations_to_boxes(boxes, priors)
        boxes = xywh2x1y1x2y2(boxes)
        # 恢复到输入图像尺寸上
        resize = target["resize"]
        boxes = boxes*torch.as_tensor([resize[1],resize[0],resize[1],resize[0]],device=self.device).unsqueeze(0)

        # 裁剪到图像内
        boxes[...,[0,2]] = torch.clamp(boxes[...,[0,2]],0,resize[1])
        boxes[...,[1,3]] = torch.clamp(boxes[...,[1,3]],0,resize[0])

        return boxes

    def apply_nms(self,prediction):
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

            if len(last_labels)==0:
                return None

            # resize 到原图上
            h_ori,w_ori = prediction["original_size"]
            h_re,w_re = prediction["resize"]
            h_ori = h_ori.float()
            w_ori = w_ori.float()

            # to pad图上
            if h_ori > w_ori:
                h_scale = h_ori/h_re
                w_scale = h_ori/w_re
                # 去除pad部分
                diff = h_ori - w_ori
                for i in range(len(last_boxes)):
                    last_boxes[i][[0,2]]*=w_scale
                    last_boxes[i][[1,3]]*=h_scale

                    last_boxes[i][0] -= diff // 2
                    last_boxes[i][2] -= diff-diff // 2

            else:
                h_scale = w_ori / h_re
                w_scale = w_ori / w_re
                diff = w_ori - h_ori
                for i in range(len(last_boxes)):
                    last_boxes[i][[0,2]]*=w_scale
                    last_boxes[i][[1,3]]*=h_scale

                    last_boxes[i][1] -= diff // 2
                    last_boxes[i][3] -= diff - diff // 2

            return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes}

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

def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def assign_priors(gt_boxes, gt_labels, priors_x1y1x2y2,
                  iou_threshold=0.5):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors_x1y1x2y2 (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # 计算IOU
    # # size: num_priors x num_targets
    ious = iou_of(gt_boxes.unsqueeze(0), priors_x1y1x2y2.unsqueeze(1))
    # print(ious.shape) # torch.Size([5625, 2])

    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    # 每个gt-box 对应一个priors-box(按IOU值最大分配)
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]

    return boxes, labels

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

def fitter_null(tmp_boxes,tmp_labels):
    box_list=[]
    label_list=[]
    for box,label in zip(tmp_boxes,tmp_labels):
        if sum(box)>0:
            box_list.append(box)
            label_list.append(label)

    return torch.stack(box_list),torch.stack(label_list)