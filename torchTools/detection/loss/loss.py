"""
try:
    from .boxestool import batched_nms
except:
    from boxestool import batched_nms
"""
try:
    from .py_cpu_nms import py_cpu_nms
except:
    from py_cpu_nms import py_cpu_nms

from torch import nn
import torch
from torch.nn import functional as F
import random


class YOLOv1Loss(nn.Module):
    def __init__(self,device="cpu",num_anchors=2,
                 num_classes=20, # 不包括背景
                 threshold_conf=0.05,
                 threshold_cls=0.5,
                 conf_thres=0.8,
                 nms_thres=0.4,
                 filter_labels:list = [],
                 mulScale=False):
        super(YOLOv1Loss, self).__init__()
        self.device = device
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.threshold_conf = threshold_conf
        self.threshold_cls = threshold_cls
        self.mulScale = mulScale
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.filter_labels = filter_labels

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
        :param preds: [n,7,7,12]
        :param targets: [n,7,7,12]
        :return:
        """
        losses = {
            "loss_conf": 0,
            "loss_no_conf": 0,
            "loss_box": 0,
            "loss_clf": 0,
            "loss_no_clf": 0,
            # "iou_loss": iou_loss
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
                targets = self.normalize((fh, fw), target_origin)

                preds = preds.contiguous().view(-1, self.num_anchors * (5 + self.num_classes))
                targets = targets.contiguous().view(-1, self.num_anchors * (5 + self.num_classes))
                index = targets[..., 4] == 1
                no_index = targets[..., 4] != 1
                has_obj = preds[index]
                no_obj = preds[no_index]
                targ_obj = targets[index]

                loss_conf = F.binary_cross_entropy(has_obj[..., 4], torch.ones_like(has_obj[..., 4]).detach(),
                                                   reduction="sum")  # 对应目标
                loss_conf += F.binary_cross_entropy(has_obj[..., 4 + 5 + self.num_classes],
                                                    torch.ones_like(has_obj[..., 4]).detach(), reduction="sum")  # 对应目标


                loss_no_conf = F.binary_cross_entropy(no_obj[..., 4], torch.zeros_like(no_obj[..., 4]).detach(),
                                                      reduction="sum")  # 对应背景
                loss_no_conf += F.binary_cross_entropy(no_obj[..., 4 + 5 + self.num_classes],
                                                       torch.zeros_like(no_obj[..., 4]).detach(), reduction="sum")  # 对应背景

                # boxes loss
                # loss_box = F.mse_loss(has_obj[...,:4],targ_obj[...,:4].detach(),reduction="sum")
                # loss_box += F.mse_loss(has_obj[...,5+self.num_classes:4+5+self.num_classes],targ_obj[...,:4].detach(),reduction="sum")

                loss_box = F.smooth_l1_loss(has_obj[..., :4], targ_obj[..., :4].detach(), reduction="sum")
                loss_box += F.smooth_l1_loss(has_obj[..., 5 + self.num_classes:4 + 5 + self.num_classes],
                                             targ_obj[..., 5 + self.num_classes:4 + 5 + self.num_classes].detach(), reduction="sum")

                # classify loss
                loss_clf = F.binary_cross_entropy(has_obj[..., 5], targ_obj[..., 5].detach(), reduction="sum")
                loss_clf += F.binary_cross_entropy(has_obj[..., 5 + 5 + self.num_classes], targ_obj[..., 5].detach(),
                                                   reduction="sum")

                loss_no_clf = F.binary_cross_entropy(no_obj[..., 5], torch.zeros_like(no_obj[..., 5]).detach(), reduction="sum")
                loss_no_clf += F.binary_cross_entropy(no_obj[..., 5 + 5 + self.num_classes],
                                                      torch.zeros_like(no_obj[..., 5]).detach(), reduction="sum")

                losses["loss_conf"] += loss_conf
                losses["loss_no_conf"] += loss_no_conf * 0.1  # 0.05
                losses["loss_box"] += loss_box * 5  # 50
                losses["loss_clf"] += loss_clf
                losses["loss_no_clf"] += loss_no_clf * 0.1  # 0.05

        return losses

    def normalize(self, featureShape, target):
        grid_ceil_h, grid_ceil_w = featureShape
        h, w = target["resize"]
        boxes = target["boxes"]
        labels = target["labels"]
        strides_h = h//grid_ceil_h
        strides_w = w//grid_ceil_w

        result = torch.zeros([1, grid_ceil_h, grid_ceil_w, # len(labels)
                              self.num_anchors * (5 + self.num_classes)],
                             dtype=boxes.dtype,
                             device=boxes.device)

        # for idx, (box, label) in enumerate(zip(boxes, labels)):
        idx = 0
        box = boxes
        label = labels
        # x1,y1,x2,y2->x0,y0,w,h
        x1 = box[:, 0]
        y1 = box[:, 1]
        x2 = box[:, 2]
        y2 = box[:, 3]

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
        x0 = (x0 - grid_ceil[0].float() * strides_w) / strides_w
        y0 = (y0 - grid_ceil[1].float() * strides_h) / strides_h

        for i, (y, x) in enumerate(zip(grid_ceil[1], grid_ceil[0])):
            result[idx, y, x, [0, 0 + 5 + self.num_classes]] = x0[i]
            result[idx, y, x, [1, 1 + 5 + self.num_classes]] = y0[i]
            result[idx, y, x, [2, 2 + 5 + self.num_classes]] = w_b[i]
            result[idx, y, x, [3, 3 + 5 + self.num_classes]] = h_b[i]
            result[idx, y, x, [4, 4 + 5 + self.num_classes]] = 1  # 置信度
            result[idx, y, x, [5, 5 + 5 + self.num_classes]] = label[i].float()+1 # 源数据传入的标签都是从0开始的，加1这样0对应 就是背景

        return result

    def predict(self, preds_list,targets_origin):
        result = []

        for idx, preds in enumerate(preds_list):
            bs, fh, fw = preds.shape[:-1]
            preds = preds.contiguous().view(-1, self.num_anchors, 5 + self.num_classes)
            # preds = preds.contiguous().view(bs,fh,fw,self.num_anchors,5 + self.num_classes)
            # 选择置信度最高的对应box(多个box时)
            """
            new_preds = torch.zeros_like(preds)[:, 0, :]
            for i, p in enumerate(preds):
                # conf
                if p[0, 4] * p[0, 5] > p[1, 4] * p[1, 5]:
                # if p[0, 4] > p[1, 4]:
                    new_preds[i] = preds[i, 0, :]
                else:
                    new_preds[i] = preds[i, 1, :]
            """
            new_preds = torch.max(preds, 1)[0]

            preds = new_preds.contiguous().view(bs, -1, 5 + self.num_classes)

            for i in range(bs):
                targets = targets_origin[i]
                pred_box = preds[i, :, :4]
                pred_conf = preds[i, :, 4]
                pred_cls = preds[i, :, 5]  # *pred_conf # # 推理时做 p_cls*confidence

                # 转成x1,y1,x2,y2
                pred_box = self.reverse_normalize((fh, fw), pred_box,targets)

                # pred_box = clip_boxes_to_image(pred_box, input_img[0].size()[-2:])  # 裁剪到图像内
                # # 过滤尺寸很小的框
                # keep = remove_small_boxes(pred_box.round(), self.min_size)
                # pred_box = pred_box[keep]
                # pred_cls = pred_cls[keep]
                # confidence = pred_conf[keep]#.squeeze(1)

                confidence = pred_conf

                # condition = (pred_cls * confidence).max(dim=0)[0] > self.threshold_conf
                # condition = (pred_cls * confidence) > self.threshold_conf
                condition = confidence > self.threshold_conf

                keep = torch.nonzero(condition).squeeze(1)
                pred_box = pred_box[keep]
                pred_cls = pred_cls[keep]
                confidence = confidence[keep]

                # labels and scores
                # scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
                # scores, labels = pred_cls.max(dim=1)

                scores, labels = pred_cls, torch.ones_like(pred_cls)

                # 过滤分类分数低的
                # keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
                keep = torch.nonzero(scores > self.threshold_cls)
                pred_box, scores, labels, confidence = pred_box[keep], scores[keep], labels[keep], confidence[keep]

                if len(result) < bs:
                    result.append({"boxes": pred_box, "scores": scores, "labels": labels, "confidence": confidence})
                    result[i].update(targets)
                else:
                    assert len(result) == bs, print("error")
                    result[i]["boxes"] = torch.cat((result[i]["boxes"], pred_box), 0)
                    result[i]["scores"] = torch.cat((result[i]["scores"], scores), 0)
                    result[i]["labels"] = torch.cat((result[i]["labels"], labels), 0)
                    result[i]["confidence"] = torch.cat((result[i]["confidence"], confidence), 0)

        return result

    def reverse_normalize(self,featureShape,boxes, target):
        # [x0,y0,w,h]-->normalize 0~1--->[x1,y1,x2,y2]

        h, w = target["resize"]
        h_f, w_f = featureShape
        strides_h = h // h_f
        strides_w = w // w_f

        # to 格网(x,y) 格式
        temp = torch.arange(0, len(boxes))
        grid_y = temp // w_f
        grid_x = temp - grid_y * w_f

        x0 = boxes[:, 0] * strides_w + (grid_x * strides_w).float().to(self.device)
        y0 = boxes[:, 1] * strides_h + (grid_y * strides_h).float().to(self.device)
        w_b = boxes[:, 2] * w
        h_b = boxes[:, 3] * h

        x1 = x0 - w_b / 2.
        y1 = y0 - h_b / 2.
        x2 = x0 + w_b / 2.
        y2 = y0 + h_b / 2.

        # 裁剪到框内
        x1 = x1.clamp(0,w)
        x2 = x2.clamp(0,w)
        y1 = y1.clamp(0,h)
        y2 = y2.clamp(0,h)

        return torch.stack((x1, y1, x2, y2), dim=0).t()

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
                    keep=py_cpu_nms(_boxes.cpu().numpy(),_scores.cpu().numpy(),self.nms_thres)
                    # keep = nms(_boxes, _scores, nms_thres)
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