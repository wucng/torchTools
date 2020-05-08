"""
# 结合SSD的思想
# yolov2,v3内置的先念框，以网格左上角为中心，预设置的w,h作为先念框的大小
# 统一都缩减到输入 图像上
# 在结合SSD的方式筛选正负样本（根据IOU计算）


try:
    from .boxestool import batched_nms
except:
    from boxestool import batched_nms
"""
from .nms_pytorch import nms,nms2

from torch import nn
import torch
from torch.nn import functional as F
import random
import numpy as np
from .focalLoss import smooth_l1_loss_jit,giou_loss_jit,ciou_loss_jit,diou_loss_jit,\
    sigmoid_focal_loss_jit,softmax_focal_loss_jit
# import math
from math import sqrt
class YOLOv1Loss(nn.Module):
    def __init__(self,cfg,device="cpu"):
        super(YOLOv1Loss, self).__init__()
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
        self.resize = cfg["work"]["train"]["resize"]

    def forward(self,preds,targets):
        if "boxes" not in targets[0]:
            # return self.predict(preds,targets)
            results = self.predict(preds,targets)
            results = [self.apply_nms(result) for result in results]
            return results
        else:
            return self.compute_loss(preds, targets)

    def compute_loss(self,preds_list, targets_origin):
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
            "loss_conf": 0,
            "loss_no_conf": 0,
            # "loss_box": 0,
            "loss_clf": 0,
            "loss_no_clf": 0,
            "loss_iou": 0
        }

        # normalize
        targets = self.normalize(targets_origin)
        pred_cls,pred_box = preds_list

        index = targets[..., 4] == 1
        no_index = targets[..., 4] != 1
        conf = pred_cls[index][...,0]
        no_conf = pred_cls[no_index][...,0]
        box = pred_box[index]
        cls = pred_cls[index][...,1:]
        no_cls = pred_cls[no_index][...,1:]

        tbox = targets[index][...,:4]
        tcls = targets[index][...,5:]

        loss_conf = sigmoid_focal_loss_jit(conf, torch.ones_like(conf).detach(),
                                           self.alpha, self.gamma, reduction="sum")
        loss_no_conf = sigmoid_focal_loss_jit(no_conf, torch.zeros_like(no_conf).detach(),
                                              self.alpha, self.gamma, reduction="sum")
        # loss_box = smooth_l1_loss_jit(torch.sigmoid(box), tbox.detach(), 2e-5, reduction="sum")
        # loss_box_xy = F.smooth_l1_loss(torch.sigmoid(box[...,:2]),tbox[...,:2], reduction="sum")
        # loss_box_wh = F.smooth_l1_loss(torch.sigmoid(box[...,2:]).sqrt(),tbox[...,2:].sqrt(), reduction="sum")
        # loss_box = loss_box_xy+loss_box_wh
        loss_clf = sigmoid_focal_loss_jit(cls, tcls.detach(),
                                          self.alpha, self.gamma, reduction="sum")
        loss_no_clf = F.mse_loss(torch.sigmoid(no_cls), torch.zeros_like(no_cls).detach(),
                                 reduction="sum")

        # iou loss
        # loss_iou = giou_loss_jit(xywh2x1y1x2y2(torch.sigmoid(box)),xywh2x1y1x2y2(tbox).detach(), reduction="sum")
        loss_iou = ciou_loss_jit(xywh2x1y1x2y2(torch.sigmoid(box)),xywh2x1y1x2y2(tbox).detach(), reduction="sum")

        lambdanoobj = 0.5
        lambdaobj = 5.0
        losses["loss_conf"] = loss_conf*lambdaobj
        losses["loss_no_conf"] = loss_no_conf * lambdanoobj
        # losses["loss_box"] = loss_box*lambdaobj
        losses["loss_clf"] = loss_clf*lambdaobj
        losses["loss_no_clf"] = loss_no_clf * lambdanoobj
        losses["loss_iou"] = loss_iou*lambdaobj

        return losses

    def normalize(self,targets):
        last_result = []
        for target in targets:
            result_list = []
            h, w = target["resize"]
            boxes = target["boxes"]
            labels = target["labels"]
            for stride in self.strides:
                grid_ceil_h, grid_ceil_w = h//stride,w//stride
                strides_h,strides_w=stride,stride

                result = torch.zeros([1, grid_ceil_h, grid_ceil_w,
                                      self.num_anchors, 5 + self.num_classes],
                                     dtype=torch.float32,
                                     device=self.device)
                idx = 0
                # x1,y1,x2,y2->x0,y0,w,h
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]

                # [x0,y0,w,h]
                x0 = (x1 + x2) / 2.
                y0 = (y1 + y2) / 2.
                w_b = (x2 - x1) / w  # 0~1
                h_b = (y2 - y1) / h  # 0~1

                # 判断box落在哪个grid ceil
                # 取格网左上角坐标
                grid_ceil = ((x0 / strides_w).int(), (y0 / strides_h).int())

                # normal 0~1
                # gt_box 中心点坐标-对应grid cell左上角的坐标/ 格网大小使得范围变成0到1
                x0 = (x0 - grid_ceil[0].float() * strides_w) / w
                y0 = (y0 - grid_ceil[1].float() * strides_h) / h

                for i, (y, x) in enumerate(zip(grid_ceil[1], grid_ceil[0])):
                    result[idx, y, x, :,0] = x0[i]
                    result[idx, y, x, :,1] = y0[i]
                    result[idx, y, x, :,2] = w_b[i]
                    result[idx, y, x, :,3] = h_b[i]
                    result[idx, y, x, :,4] = 1  # 置信度
                    result[idx, y, x, :,5+int(labels[i])] = 1 # 转成one-hot

                result_list.append(result.view(1,-1,5 + self.num_classes))
            last_result.append(torch.cat(result_list,1))
        return torch.cat(last_result,0)

    def predict(self, preds_list,targets_origin):
        """
        :param preds_list:
                   #（2个特征为例,batch=2）
                   preds_list=[(2,28,28,12),(2,14,14,12)]
        :param targets_origin:
                  [{"resize":(h,w),"origin_size":(h,w)},{"resize":(h,w),"origin_size":(h,w)}]
        :return:
        """
        pred_clses, pred_boxes = preds_list
        pred_clses, pred_boxes = torch.sigmoid(pred_clses),torch.sigmoid(pred_boxes)
        pred_boxes = self.reverse_normalize(pred_boxes)
        result = []

        for i,(pred_cls,pred_box) in enumerate(zip(pred_clses,pred_boxes)):
            confidence = pred_cls[...,0]
            pred_cls = pred_cls[...,1:]

            condition = (pred_cls * confidence).max(dim=0)[0] > self.threshold_conf
            keep = torch.nonzero(condition).squeeze(1)

            if len(keep)==0:
                pred_box = torch.zeros([1,4],dtype=pred_box.dtype,device=pred_box.device)
                scores = torch.zeros([1,],dtype=pred_box.dtype,device=pred_box.device)
                labels = torch.zeros([1,],dtype=torch.long,device=pred_box.device)
                confidence = torch.zeros([1,],dtype=pred_box.dtype,device=pred_box.device)

            else:
                pred_box = pred_box[keep]
                pred_cls = pred_cls[keep]
                confidence = confidence[keep]

                # labels and scores
                # scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
                scores, labels = pred_cls.max(dim=1)

                # 过滤分类分数低的
                # keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
                keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
                if len(keep)==0:
                    pred_box = torch.zeros([1, 4], dtype=pred_box.dtype, device=pred_box.device)
                    scores = torch.zeros([1, ], dtype=pred_box.dtype, device=pred_box.device)
                    labels = torch.zeros([1, ], dtype=torch.long, device=pred_box.device)
                    confidence = torch.zeros([1, ], dtype=pred_box.dtype, device=pred_box.device)
                else:
                    pred_box, scores, labels, confidence = pred_box[keep], scores[keep], labels[keep], confidence[keep]

            result.append({"boxes": pred_box, "scores": scores, "labels": labels, "confidence": confidence})
            result[i].update(targets_origin[i])

        return result

    def reverse_normalize(self,pboxes):
        # [x0,y0,w,h]-->normalize 0~1--->[x1,y1,x2,y2]
        bs = pboxes.size(0)
        pboxes = pboxes.view(bs,-1,self.num_anchors,4)
        for i,cboxes in enumerate(pboxes):
            index = 0
            h, w = self.resize
            for stride in self.strides:
                strides_h, strides_w = stride, stride
                h_f,w_f = h//strides_h, w//strides_w
                boxes = cboxes[index:index+h_f*w_f,...]
                # to 格网(x,y) 格式
                temp = torch.arange(0, len(boxes),device=self.device)
                grid_y = temp // w_f
                grid_x = temp - grid_y * w_f

                for j in range(self.num_anchors):
                    x0 = boxes[:,j, 0] * w + (grid_x * strides_w).float()
                    y0 = boxes[:,j, 1] * h + (grid_y * strides_h).float()
                    w_b = boxes[:,j, 2] * w
                    h_b = boxes[:,j, 3] * h

                    x1 = x0 - w_b / 2.
                    y1 = y0 - h_b / 2.
                    x2 = x0 + w_b / 2.
                    y2 = y0 + h_b / 2.

                    # 裁剪到框内
                    x1 = x1.clamp(0,w)
                    x2 = x2.clamp(0,w)
                    y1 = y1.clamp(0,h)
                    y2 = y2.clamp(0,h)

                    boxes[:, j, 0] = x1
                    boxes[:, j, 1] = y1
                    boxes[:, j, 2] = x2
                    boxes[:, j, 3] = y2

                pboxes[i,index:index+h_f*w_f,...] = boxes
                index += h_f*w_f

        return pboxes.view(bs,-1,4)

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

class YOLOv2Loss(nn.Module):
    def __init__(self,cfg, device="cpu"):
        super(YOLOv2Loss,self).__init__()
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
        scale = 1.0*min(self.resize)/self.min_dim
        self.min_sizes = [int(c*scale) for c in cfg["network"]["prioriBox"]["min_sizes"]]
        self.max_sizes = [int(c*scale) for c in cfg["network"]["prioriBox"]["max_sizes"]]
        self.aspect_ratios = cfg["network"]["prioriBox"]["aspect_ratios"]
        self.variance = cfg["network"]["prioriBox"]["variance"]
        self.clip = cfg["network"]["prioriBox"]["clip"]
        self.thred_iou = cfg["network"]["prioriBox"]["thred_iou"]

        self.priorBoxWH = self.get_prior_box_wh()

    
    def get_prior_box_wh(self):
        priors = []
        h, w = self.resize
        for idx, stride in enumerate(self.strides):
            # fh, fw = h // stride, w // stride
            # small sized square box
            size_min = self.min_sizes[idx]
            size_max = self.max_sizes[idx]
            size = size_min
            bh, bw = 1.0 * size / h, 1.0 * size / w
            priors.append([bw, bh])

            # big sized square box
            size = sqrt(size_min * size_max)
            bh, bw = 1.0 * size / h, 1.0 * size / w
            priors.append([bw, bh])

            # change h/w ratio of the small sized box
            size = size_min
            bh, bw = 1.0 * size / h, 1.0 * size / w
            for ratio in self.aspect_ratios[idx]:
                ratio = sqrt(ratio)
                priors.append([bw * ratio, 1.0 * bh / ratio])
                priors.append([1.0 * bw / ratio, bh * ratio])

        priors = torch.tensor(priors, device=self.device)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors

    def forward(self,preds,targets):
        if "boxes" not in targets[0]:
            # return self.predict(preds,targets)
            results = self.predict(preds,targets)
            results = [self.apply_nms(result) for result in results]
            return results
        else:
            if self.method:
                return self.compute_loss(preds, targets) # 推荐
            else:
                return self.compute_loss2(preds, targets)

    def compute_loss(self,preds_list, targets_origin):
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
            "loss_conf": 0,
            "loss_no_conf": 0,
            "loss_box": 0,
            "loss_clf": 0,
            "loss_no_clf": 0#,
            # "loss_iou": 0
        }

        # normalize
        targets = self.normalize(targets_origin)
        pred_cls, pred_box = preds_list

        index = targets[..., 4] > 0
        no_index = targets[..., 4] == 0
        conf = pred_cls[index][..., 0]
        no_conf = pred_cls[no_index][..., 0]
        box = pred_box[index]
        cls = pred_cls[index][..., 1:]
        no_cls = pred_cls[no_index][..., 1:]

        tbox = targets[index][..., :4]
        tcls = targets[index][..., 5:]
        weight_iou = targets[index][..., 4]

        loss_conf = sigmoid_focal_loss_jit(conf, torch.ones_like(conf).detach(),
                                           self.alpha, self.gamma, reduction="None")
        loss_no_conf = sigmoid_focal_loss_jit(no_conf, torch.zeros_like(no_conf).detach(),
                                              self.alpha, self.gamma, reduction="sum")
        loss_box = smooth_l1_loss_jit(box, tbox.detach(), 2e-5, reduction="None")
        loss_clf = sigmoid_focal_loss_jit(cls, tcls.detach(),
                                          self.alpha, self.gamma, reduction="None")
        loss_no_clf = F.mse_loss(torch.sigmoid(no_cls), torch.zeros_like(no_cls).detach(),
                                 reduction="sum")

        # iou loss
        # loss_iou = giou_loss_jit(xywh2x1y1x2y2(self.reverse_normalize(pred_box)[index]),
        #                          xywh2x1y1x2y2(self.reverse_normalize(targets[..., :4])[index]).detach(), reduction="None")

        # 加上权重
        loss_conf = torch.sum(loss_conf*weight_iou)
        loss_box = torch.sum(loss_box*weight_iou.unsqueeze(-1))
        loss_clf = torch.sum(loss_clf*weight_iou)
        # loss_iou = torch.sum(loss_iou*weight_iou.unsqueeze(-1))


        losses["loss_conf"] += loss_conf
        losses["loss_no_conf"] += loss_no_conf * 0.05  # 0.05
        losses["loss_box"] += loss_box * 10.   # 50
        losses["loss_clf"] += loss_clf
        losses["loss_no_clf"] += loss_no_clf * 0.05
        # losses["loss_iou"] += loss_iou * 10.0

        return losses

    def normalize(self, targets):
        last_result = []
        for target in targets:
            result_list = []
            h, w = target["resize"]
            boxes = target["boxes"]
            labels = target["labels"]
            index = 0
            for stride in self.strides:
                grid_ceil_h, grid_ceil_w = h // stride, w // stride
                strides_h, strides_w = stride, stride
                result = torch.zeros([1, grid_ceil_h, grid_ceil_w,
                                      self.num_anchors, 5 + self.num_classes],
                                     dtype=torch.float32,
                                     device=self.device)

                idx = 0
                # x1,y1,x2,y2->x0,y0,w,h
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]

                # [x0,y0,w,h]
                x0 = (x1 + x2) / 2.
                y0 = (y1 + y2) / 2.
                w_b = (x2 - x1) / w
                h_b = (y2 - y1) / h

                # 判断box落在哪个grid ceil
                # 取格网左上角坐标
                grid_ceil = ((x0 / strides_w).int(), (y0 / strides_h).int())

                # normal 0~1
                # gt_box 中心点坐标-对应grid cell左上角的坐标/ 格网大小使得范围变成0到1
                x0 = (x0 - grid_ceil[0].float() * strides_w) / w
                y0 = (y0 - grid_ceil[1].float() * strides_h) / h

                # 找到与哪个先验框的IOU最大
                gt_boxes = boxes / torch.as_tensor([w, h, w, h],dtype=torch.float32, device=self.device).unsqueeze(0) # [x1,y1,x2,y2]

                xy = torch.stack((grid_ceil[0].float() * strides_w / w,grid_ceil[1].float() * strides_h / h),-1)

                anchors_wh = self.priorBoxWH[index:index+self.num_anchors,...]
                index += self.num_anchors

                for i, (y, x) in enumerate(zip(grid_ceil[1], grid_ceil[0])):
                    anchors = torch.cat((xy[i].repeat(self.num_anchors).view(self.num_anchors,2), anchors_wh), -1)
                    # to x1y1x2y2
                    anchors = xywh2x1y1x2y2(anchors)
                    iou = box_iou(anchors, gt_boxes[i][None, :])
                    best_iou, best_anchor = iou.max(dim=0)
                    for j in range(self.num_anchors):  # 对应到每个先念框
                        # 计算对应先念框的 h与w
                        pw, ph = anchors_wh[j]
                        result[idx, y, x, j, 0] = x0[i]/self.variance[0]
                        result[idx, y, x, j, 1] = y0[i]/self.variance[0]
                        result[idx, y, x, j, 2] = torch.log(w_b[i] / pw)/self.variance[1]
                        result[idx, y, x, j, 3] = torch.log(h_b[i] / ph)/self.variance[1]
                        result[idx, y, x, j, 4] = max(best_iou,0.5) if j==best_anchor else iou[j] # 置信度(用于设置权重)
                        result[idx, y, x, j, 5 + int(labels[i])] = 1  # 转成one-hot

                result_list.append(result.view(1, -1, 5 + self.num_classes))

            last_result.append(torch.cat(result_list, 1))
        return torch.cat(last_result, 0)

    def compute_loss2(self,preds_list, targets_origin):
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
            "loss_conf": 0,
            "loss_no_conf": 0,
            "loss_box": 0,
            "loss_clf": 0,
            "loss_no_clf": 0#,
            # "loss_iou": 0
        }

        # normalize
        targets = self.normalize2(targets_origin)
        pred_cls, pred_box = preds_list

        index = targets[..., 4] == 1
        no_index = targets[..., 4] != 1
        conf = pred_cls[index][..., 0]
        no_conf = pred_cls[no_index][..., 0]
        box = pred_box[index]
        cls = pred_cls[index][..., 1:]
        no_cls = pred_cls[no_index][..., 1:]

        tbox = targets[index][..., :4]
        tcls = targets[index][..., 5:]

        loss_conf = sigmoid_focal_loss_jit(conf, torch.ones_like(conf).detach(),
                                           self.alpha, self.gamma, reduction="sum")
        loss_no_conf = sigmoid_focal_loss_jit(no_conf, torch.zeros_like(no_conf).detach(),
                                              self.alpha, self.gamma, reduction="sum")
        loss_box = smooth_l1_loss_jit(box, tbox.detach(), 2e-5, reduction="sum")
        loss_clf = sigmoid_focal_loss_jit(cls, tcls.detach(),
                                          self.alpha, self.gamma, reduction="sum")
        loss_no_clf = F.mse_loss(torch.sigmoid(no_cls), torch.zeros_like(no_cls).detach(),
                                 reduction="sum")

        # iou loss
        # loss_iou = giou_loss_jit(xywh2x1y1x2y2(self.reverse_normalize(pred_box)[index]),
        #                          xywh2x1y1x2y2(self.reverse_normalize(targets[..., :4])[index]).detach(), reduction="sum")


        losses["loss_conf"] += loss_conf
        losses["loss_no_conf"] += loss_no_conf * 0.05  # 0.05
        losses["loss_box"] += loss_box*10.0   # 50
        losses["loss_clf"] += loss_clf
        losses["loss_no_clf"] += loss_no_clf * 0.05
        # losses["loss_iou"] += loss_iou * 10.0

        return losses

    def normalize2(self, targets):
        last_result = []
        for target in targets:
            result_list = []
            h, w = target["resize"]
            boxes = target["boxes"]
            labels = target["labels"]
            index = 0
            for stride in self.strides:
                grid_ceil_h, grid_ceil_w = h // stride, w // stride
                strides_h, strides_w = stride, stride
                result = torch.zeros([1, grid_ceil_h, grid_ceil_w,
                                      self.num_anchors, 5 + self.num_classes],
                                     dtype=torch.float32,
                                     device=self.device)

                idx = 0
                # x1,y1,x2,y2->x0,y0,w,h
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]

                # [x0,y0,w,h]
                x0 = (x1 + x2) / 2.
                y0 = (y1 + y2) / 2.
                w_b = (x2 - x1) / w
                h_b = (y2 - y1) / h

                # 判断box落在哪个grid ceil
                # 取格网左上角坐标
                grid_ceil = ((x0 / strides_w).int(), (y0 / strides_h).int())

                # normal 0~1
                # gt_box 中心点坐标-对应grid cell左上角的坐标/ 格网大小使得范围变成0到1
                x0 = (x0 - grid_ceil[0].float() * strides_w) / w
                y0 = (y0 - grid_ceil[1].float() * strides_h) / h

                # 找到与哪个先验框的IOU最大
                gt_boxes = boxes / torch.as_tensor([w, h, w, h],dtype=torch.float32, device=self.device).unsqueeze(0) # [x1,y1,x2,y2]

                xy = torch.stack((grid_ceil[0].float() * strides_w / w,grid_ceil[1].float() * strides_h / h),-1)

                anchors_wh = self.priorBoxWH[index:index+self.num_anchors,...]
                index += self.num_anchors

                for i, (y, x) in enumerate(zip(grid_ceil[1], grid_ceil[0])):
                    anchors = torch.cat((xy[i].repeat(self.num_anchors).view(self.num_anchors,2), anchors_wh), -1)
                    # to x1y1x2y2
                    anchors = xywh2x1y1x2y2(anchors)
                    iou = box_iou(anchors, gt_boxes[i][None, :])
                    best_iou, best_anchor = iou.max(dim=0)
                    for j in range(self.num_anchors):  # 对应到每个先念框
                        if iou[j] > self.thred_iou or j == best_anchor:
                            # 计算对应先念框的 h与w
                            pw, ph = anchors_wh[j]
                            result[idx, y, x, j, 0] = x0[i]/self.variance[0]
                            result[idx, y, x, j, 1] = y0[i]/self.variance[0]
                            result[idx, y, x, j, 2] = torch.log(w_b[i] / pw)/self.variance[1]
                            result[idx, y, x, j, 3] = torch.log(h_b[i] / ph)/self.variance[1]
                            result[idx, y, x, j, 4] = 1 # 置信度(用于设置权重)
                            result[idx, y, x, j, 5 + int(labels[i])] = 1  # 转成one-hot

                result_list.append(result.view(1, -1, 5 + self.num_classes))

            last_result.append(torch.cat(result_list, 1))
        return torch.cat(last_result, 0)

    def predict(self, preds_list,targets_origin):
        """
        :param preds_list:
                   #（2个特征为例,batch=2）
                   preds_list=[(2,28,28,12),(2,14,14,12)]
        :param targets_origin:
                  [{"resize":(h,w),"origin_size":(h,w)},{"resize":(h,w),"origin_size":(h,w)}]
        :return:
        """
        pred_clses, pred_boxes = preds_list
        # pred_clses, pred_boxes = torch.sigmoid(pred_clses),torch.sigmoid(pred_boxes)
        pred_clses = torch.sigmoid(pred_clses)
        pred_boxes = self.reverse_normalize(pred_boxes)
        result = []

        for i,(pred_cls,pred_box) in enumerate(zip(pred_clses,pred_boxes)):
            confidence = pred_cls[...,0]
            pred_cls = pred_cls[...,1:]

            condition = (pred_cls * confidence).max(dim=0)[0] > self.threshold_conf
            keep = torch.nonzero(condition).squeeze(1)

            if len(keep)==0:
                pred_box = torch.zeros([1,4],dtype=pred_box.dtype,device=pred_box.device)
                scores = torch.zeros([1,],dtype=pred_box.dtype,device=pred_box.device)
                labels = torch.zeros([1,],dtype=torch.long,device=pred_box.device)
                confidence = torch.zeros([1,],dtype=pred_box.dtype,device=pred_box.device)

            else:
                pred_box = pred_box[keep]
                pred_cls = pred_cls[keep]
                confidence = confidence[keep]

                # labels and scores
                # scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
                scores, labels = pred_cls.max(dim=1)

                # 过滤分类分数低的
                # keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
                keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
                if len(keep)==0:
                    pred_box = torch.zeros([1, 4], dtype=pred_box.dtype, device=pred_box.device)
                    scores = torch.zeros([1, ], dtype=pred_box.dtype, device=pred_box.device)
                    labels = torch.zeros([1, ], dtype=torch.long, device=pred_box.device)
                    confidence = torch.zeros([1, ], dtype=pred_box.dtype, device=pred_box.device)
                else:
                    pred_box, scores, labels, confidence = pred_box[keep], scores[keep], labels[keep], confidence[keep]

            result.append({"boxes": pred_box, "scores": scores, "labels": labels, "confidence": confidence})
            result[i].update(targets_origin[i])

        return result

    def reverse_normalize(self,pboxes):
        # [x0,y0,w,h]-->normalize 0~1--->[x1,y1,x2,y2]
        bs = pboxes.size(0)
        pboxes = pboxes.view(bs,-1,self.num_anchors,4)
        for i,cboxes in enumerate(pboxes):
            h, w = self.resize
            index = 0
            idx = 0
            for stride in self.strides:
                strides_h, strides_w = stride, stride
                h_f,w_f = h//strides_h, w//strides_w
                boxes = cboxes[index:index+h_f*w_f,...]
                # to 格网(x,y) 格式
                temp = torch.arange(0, len(boxes))
                grid_y = temp // w_f
                grid_x = temp - grid_y * w_f

                anchors_wh = self.priorBoxWH[idx:idx + self.num_anchors, ...]
                idx += self.num_anchors

                for j in range(self.num_anchors):
                    pw,ph = anchors_wh[j]

                    x0 = boxes[:, j, 0] * w*self.variance[0] + (grid_x * strides_w).float().to(self.device)
                    y0 = boxes[:, j, 1] * h*self.variance[0] + (grid_y * strides_h).float().to(self.device)
                    w_b = torch.exp(boxes[:, j, 2]*self.variance[1]) * pw * w
                    h_b = torch.exp(boxes[:, j, 3]*self.variance[1]) * ph * h

                    x1 = x0 - w_b / 2.
                    y1 = y0 - h_b / 2.
                    x2 = x0 + w_b / 2.
                    y2 = y0 + h_b / 2.

                    # 裁剪到框内
                    x1 = x1.clamp(0,w)
                    x2 = x2.clamp(0,w)
                    y1 = y1.clamp(0,h)
                    y2 = y2.clamp(0,h)

                    boxes[:, j, 0] = x1
                    boxes[:, j, 1] = y1
                    boxes[:, j, 2] = x2
                    boxes[:, j, 3] = y2

                pboxes[i,index:index+h_f*w_f,...] = boxes
                index += h_f*w_f

        return pboxes.view(bs,-1,4)

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