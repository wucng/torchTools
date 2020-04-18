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
import numpy as np


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
            return self.compute_loss(preds, targets,useFocal=True)

    def compute_loss(self,preds_list, targets_origin,useFocal=False,alpha=1.0,gamma=2):
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

                # preds = preds.contiguous().view(-1, self.num_anchors * (5 + self.num_classes))
                # targets = targets.contiguous().view(-1, self.num_anchors * (5 + self.num_classes))

                preds = preds.contiguous().view(-1,5 + self.num_classes)
                targets = targets.contiguous().view(-1, 5 + self.num_classes)

                index = targets[..., 4] == 1
                no_index = targets[..., 4] != 1
                has_obj = preds[index]
                no_obj = preds[no_index]
                targ_obj = targets[index]

                loss_conf = F.binary_cross_entropy(has_obj[..., 4], torch.ones_like(has_obj[..., 4]).detach(),
                                                   reduction="sum")  # 对应目标

                loss_no_conf = F.binary_cross_entropy(no_obj[..., 4], torch.zeros_like(no_obj[..., 4]).detach(),
                                                      reduction="sum")  # 对应背景
                # boxes loss
                # loss_box = F.mse_loss(has_obj[...,:4],targ_obj[...,:4].detach(),reduction="sum")
                loss_box = F.smooth_l1_loss(has_obj[..., :4], targ_obj[..., :4].detach(), reduction="sum")

                # classify loss
                # loss_clf = F.mse_loss(has_obj[..., 5:], targ_obj[..., 5:].detach(), reduction="sum")
                loss_clf = F.binary_cross_entropy(has_obj[..., 5:], targ_obj[..., 5:].detach(), reduction="sum")
                # loss_clf = F.cross_entropy(has_obj[..., 5:], targ_obj[..., 5:].argmax(-1), reduction="sum")

                # no obj classify loss
                loss_no_clf = F.mse_loss(no_obj[..., 5:], torch.zeros_like(no_obj[..., 5:]).detach(), reduction="sum")
                # loss_no_clf = F.binary_cross_entropy(no_obj[..., 5:], torch.zeros_like(no_obj[..., 5:]).detach(), reduction="sum")

                if useFocal:
                    loss_conf = alpha * (1 - torch.exp(-loss_conf)) ** gamma * loss_conf
                    loss_no_conf = alpha * (1 - torch.exp(-loss_no_conf)) ** gamma * loss_no_conf
                    # loss_box = alpha * (1 - torch.exp(-loss_box)) ** gamma * loss_box
                    loss_clf = alpha * (1 - torch.exp(-loss_clf)) ** gamma * loss_clf
                    loss_no_clf = alpha * (1 - torch.exp(-loss_no_clf)) ** gamma * loss_no_clf


                losses["loss_conf"] += loss_conf
                losses["loss_no_conf"] += loss_no_conf * 0.05  # 0.05
                losses["loss_box"] += loss_box * 50.  # 50
                losses["loss_clf"] += loss_clf
                losses["loss_no_clf"] += loss_no_clf * 0.05

        return losses

    def normalize(self, featureShape, target):
        grid_ceil_h, grid_ceil_w = featureShape
        h, w = target["resize"]
        boxes = target["boxes"]
        labels = target["labels"]
        strides_h = h//grid_ceil_h
        strides_w = w//grid_ceil_w

        result = torch.zeros([1, grid_ceil_h, grid_ceil_w, # len(labels)
                              self.num_anchors, 5 + self.num_classes],
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
            result[idx, y, x, :,0] = x0[i]
            result[idx, y, x, :,1] = y0[i]
            result[idx, y, x, :,2] = w_b[i]
            result[idx, y, x, :,3] = h_b[i]
            result[idx, y, x, :,4] = 1  # 置信度
            result[idx, y, x, :,5+int(label[i])] = 1 # 转成one-hot

        return result

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
            preds = preds.contiguous().view(bs,-1, self.num_anchors, 5 + self.num_classes)

            for i in range(bs):
                targets = targets_origin[i]
                new_preds = preds[i]
                new_preds[...,:4] = self.reverse_normalize((fh, fw), new_preds[...,:4], targets)
                new_preds = new_preds[:, (new_preds[:, :, 4] * new_preds[:, :, 5:].max(-1)[0]).max(1)[1], :][:, 0, :]
                # new_preds = new_preds[:, new_preds[:, :, 4].max(1)[1], :][:, 0, :]
                # new_preds = new_preds[:, new_preds[:, :, 5:].max(-1)[0].max(1)[1], :][:, 0, :]

                pred_box = new_preds[:,:4]
                pred_conf = new_preds[:, 4]
                pred_cls = new_preds[:, 5:]

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

                if len(keep)==0:
                    pred_box = torch.zeros([1,4],dtype=pred_box.dtype,device=pred_box.device)
                    scores = torch.zeros([1,1],dtype=pred_box.dtype,device=pred_box.device)
                    labels = torch.zeros([1,1],dtype=pred_box.dtype,device=pred_box.device)
                    confidence = torch.zeros([1,1],dtype=pred_box.dtype,device=pred_box.device)

                else:
                    pred_box = pred_box[keep]
                    pred_cls = pred_cls[keep]
                    confidence = confidence[keep]

                    # labels and scores
                    # scores, labels = torch.softmax(pred_cls, -1).max(dim=1)
                    scores, labels = pred_cls.max(dim=1)

                    # 过滤分类分数低的
                    # keep = torch.nonzero(scores > self.threshold_cls).squeeze(1)
                    keep = torch.nonzero(scores > self.threshold_cls)
                    if len(keep)==0:
                        pred_box = torch.zeros([1, 4], dtype=pred_box.dtype, device=pred_box.device)
                        scores = torch.zeros([1, 1], dtype=pred_box.dtype, device=pred_box.device)
                        labels = torch.zeros([1, 1], dtype=pred_box.dtype, device=pred_box.device)
                        confidence = torch.zeros([1, 1], dtype=pred_box.dtype, device=pred_box.device)
                    else:
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

        for j in range(self.num_anchors):
            x0 = boxes[:,j, 0] * strides_w + (grid_x * strides_w).float().to(self.device)
            y0 = boxes[:,j, 1] * strides_h + (grid_y * strides_h).float().to(self.device)
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

class YOLOv2Loss(YOLOv1Loss):
    """
    5个先验框的width和height:(输入大小=416，stride=32，对应的先念框)
    COCO: (0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)
    VOC: (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)
    """
    # w,h
    PreBoxSize = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]
    PreFSize = 416//32
    # PreStride = 32
    # PreSize = 416

    def __init__(self, device="cpu", num_anchors=5,
                 num_classes=20,  # 不包括背景
                 threshold_conf=0.05,
                 threshold_cls=0.5,
                 conf_thres=0.8,
                 nms_thres=0.4,
                 filter_labels: list = [],
                 mulScale=False,):
        super(YOLOv2Loss,self).__init__(device, num_anchors,
                                        num_classes,threshold_conf,threshold_cls,
                                        conf_thres,nms_thres,filter_labels,mulScale)

        self.PreBoxSize = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                           (11.2364, 10.0071)]
        self.PreFSize = 416 // 32

        # assert num_anchors==len(self.PreBoxSize),print("num_anchors:%d not equal num of PreBoxSize"%(num_anchors))

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self,preds,targets,lossfunc="v1"):
        if "boxes" not in targets[0]:
            # return self.predict(preds,targets)
            results = self.predict(preds,targets)
            results = [self.apply_nms(result) for result in results]
            return results
        else:
            if lossfunc=="v1": # 类似于 yolov1的方式
                return self.compute_loss(preds, targets,useFocal=True)
            else:
                return self.compute_loss2(preds, targets,useFocal=True)


    def normalize(self, featureShape, target):
        """不做筛选所有的anchor都参与计算"""
        grid_ceil_h, grid_ceil_w = featureShape
        h, w = target["resize"]
        boxes = target["boxes"]
        labels = target["labels"]
        strides_h = h // grid_ceil_h
        strides_w = w // grid_ceil_w

        result = torch.zeros([1, grid_ceil_h, grid_ceil_w,  # len(labels)
                              self.num_anchors, 5 + self.num_classes],
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
        # w_b = (x2 - x1) / w  # 0~1
        # h_b = (y2 - y1) / h  # 0~1
        w_b = (x2 - x1) / strides_w
        h_b = (y2 - y1) / strides_h

        # 判断box落在哪个grid ceil
        # 取格网左上角坐标
        grid_ceil = ((x0 / strides_w).int(), (y0 / strides_h).int())

        # normal 0~1
        # gt_box 中心点坐标-对应grid cell左上角的坐标/ 格网大小使得范围变成0到1
        x0 = (x0 - grid_ceil[0].float() * strides_w) / strides_w
        y0 = (y0 - grid_ceil[1].float() * strides_h) / strides_h


        for i, (y, x) in enumerate(zip(grid_ceil[1], grid_ceil[0])):
            for j in range(self.num_anchors):  # 对应到每个先念框
                # 计算对应先念框的 h与w
                pw, ph = self.PreBoxSize[j]
                pw *= (grid_ceil_w / self.PreFSize)
                ph *= (grid_ceil_h / self.PreFSize)

                result[idx, y, x, j, 0] = x0[i]
                result[idx, y, x, j, 1] = y0[i]
                result[idx, y, x, j, 2] = torch.log(w_b[i]/pw)
                result[idx, y, x, j, 3] = torch.log(h_b[i]/ph)
                result[idx, y, x, j, 4] = 1  # 置信度
                result[idx, y, x, j, 5 + int(label[i])] = 1  # 转成one-hot

        return result

    def compute_loss2(self,preds_list, targets_origin,useFocal=False,alpha=1.0,gamma=2):
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
                targets,noobj_mask = self.normalize2((fh, fw), target_origin)

                # preds = preds.contiguous().view(-1, self.num_anchors * (5 + self.num_classes))
                # targets = targets.contiguous().view(-1, self.num_anchors * (5 + self.num_classes))

                preds = preds.contiguous().view(-1,5 + self.num_classes)
                targets = targets.contiguous().view(-1, 5 + self.num_classes)
                noobj_mask = noobj_mask.contiguous().view(-1,1)

                mask = targets[..., 4].unsqueeze(-1)
                tconf = targets[..., 4].unsqueeze(-1)
                tx = targets[..., 0].unsqueeze(-1)
                ty = targets[..., 1].unsqueeze(-1)
                tw = targets[..., 2].unsqueeze(-1)
                th = targets[..., 3].unsqueeze(-1)
                tcls = targets[..., 5:]

                conf = preds[..., 4].unsqueeze(-1)
                x = preds[..., 0].unsqueeze(-1)
                y = preds[..., 1].unsqueeze(-1)
                w = preds[..., 2].unsqueeze(-1)
                h = preds[..., 3].unsqueeze(-1)
                cls = preds[..., 5:]

                loss_conf = self.bce_loss(conf * mask,torch.ones_like(conf * mask))  # 对应目标

                loss_no_conf = self.bce_loss(conf*noobj_mask,torch.zeros_like(conf*noobj_mask)) # 对应背景
                # boxes loss
                # loss_x = self.bce_loss(x * mask, tx * mask)
                # loss_y = self.bce_loss(y * mask, ty * mask)
                loss_x = self.mse_loss(x * mask, tx * mask)
                loss_y = self.mse_loss(y * mask, ty * mask)
                loss_w = self.mse_loss(w * mask, tw * mask)
                loss_h = self.mse_loss(h * mask, th * mask)
                loss_box = loss_x+loss_y+loss_w+loss_h

                # classify loss
                loss_clf = self.bce_loss(cls*mask,tcls*mask)

                # no obj classify loss
                loss_no_clf = self.mse_loss(cls*noobj_mask,torch.zeros_like(cls*noobj_mask))

                if useFocal:
                    loss_conf = alpha * (1 - torch.exp(-loss_conf)) ** gamma * loss_conf
                    loss_no_conf = alpha * (1 - torch.exp(-loss_no_conf)) ** gamma * loss_no_conf
                    # loss_box = alpha * (1 - torch.exp(-loss_box)) ** gamma * loss_box
                    loss_clf = alpha * (1 - torch.exp(-loss_clf)) ** gamma * loss_clf
                    loss_no_clf = alpha * (1 - torch.exp(-loss_no_clf)) ** gamma * loss_no_clf


                losses["loss_conf"] += loss_conf
                losses["loss_no_conf"] += loss_no_conf * 0.05  # 0.05
                losses["loss_box"] += loss_box * 50.  # 50
                losses["loss_clf"] += loss_clf
                losses["loss_no_clf"] += loss_no_clf * 0.05

        return losses

    def normalize2(self, featureShape, target):
        """加入按IOU筛选最好的anchor"""
        grid_ceil_h, grid_ceil_w = featureShape
        h, w = target["resize"]
        boxes = target["boxes"]
        labels = target["labels"]
        strides_h = h // grid_ceil_h
        strides_w = w // grid_ceil_w

        result = torch.zeros([1, grid_ceil_h, grid_ceil_w,  # len(labels)
                              self.num_anchors, 5 + self.num_classes],
                             dtype=boxes.dtype,
                             device=boxes.device)

        noobj_mask = torch.ones([1,grid_ceil_h, grid_ceil_w,self.num_anchors,1],dtype=torch.long,device=boxes.device)

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
        # w_b = (x2 - x1) / w  # 0~1
        # h_b = (y2 - y1) / h  # 0~1
        w_b = (x2 - x1) / strides_w
        h_b = (y2 - y1) / strides_h

        # 判断box落在哪个grid ceil
        # 取格网左上角坐标
        grid_ceil = ((x0 / strides_w).int(), (y0 / strides_h).int())

        # normal 0~1
        # gt_box 中心点坐标-对应grid cell左上角的坐标/ 格网大小使得范围变成0到1
        x0 = (x0 - grid_ceil[0].float() * strides_w) / strides_w
        y0 = (y0 - grid_ceil[1].float() * strides_h) / strides_h

        # 找到与哪个先验框的IOU最大
        gt_boxes = torch.stack((w_b,h_b),1)
        anchors = torch.as_tensor(self.PreBoxSize, device=self.device,dtype=boxes.dtype)
        temp_anchors = torch.cat((torch.zeros_like(anchors), anchors), -1)
        temp_gt_boxes = torch.cat((torch.zeros_like(gt_boxes), gt_boxes), -1)

        # IOU阈值
        iou_thres = 0.5

        for i, (y, x) in enumerate(zip(grid_ceil[1], grid_ceil[0])):
            # 按IOU筛选最好的anchor
            iou = box_iou(temp_anchors, temp_gt_boxes[i][None,:])
            best_iou, best_anchor = iou.max(dim=0)

            noobj_mask[idx,y,x,torch.nonzero(iou>iou_thres)[:,0],0] = 0 # 忽略 ,是目标而不是背景

            # if best_iou > 0:
            # 计算对应先念框的 h与w
            pw, ph = self.PreBoxSize[best_anchor]
            pw *= (grid_ceil_w / self.PreFSize)
            ph *= (grid_ceil_h / self.PreFSize)

            result[idx, y, x, best_anchor, 0] = x0[i]
            result[idx, y, x, best_anchor, 1] = y0[i]
            result[idx, y, x, best_anchor, 2] = torch.log(w_b[i]/pw)
            result[idx, y, x, best_anchor, 3] = torch.log(h_b[i]/ph)
            result[idx, y, x, best_anchor, 4] = 1  # 置信度
            result[idx, y, x, best_anchor, 5 + int(label[i])] = 1  # 转成one-hot

        return result,noobj_mask

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

        for j in range(self.num_anchors):
            pw, ph = self.PreBoxSize[j]
            pw *= (w_f / self.PreFSize)
            ph *= (h_f / self.PreFSize)

            x0 = boxes[:,j, 0] * strides_w + (grid_x * strides_w).float().to(self.device)
            y0 = boxes[:,j, 1] * strides_h + (grid_y * strides_h).float().to(self.device)
            w_b = torch.exp(boxes[:,j, 2]) * pw*strides_w
            h_b = torch.exp(boxes[:,j, 3]) * ph*strides_h

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

        return boxes


class YOLOv3Loss(YOLOv2Loss):
    def __init__(self, device="cpu", num_anchors=3,
                 num_classes=20,  # 不包括背景
                 threshold_conf=0.05,
                 threshold_cls=0.5,
                 conf_thres=0.8,
                 nms_thres=0.4,
                 filter_labels: list = [],
                 mulScale=False,):
        super(YOLOv3Loss,self).__init__(device, num_anchors,
                                        num_classes,threshold_conf,threshold_cls,
                                        conf_thres,nms_thres,filter_labels,mulScale)

        self.PreBoxSize = np.asarray([(116, 90), (156, 198), (373 , 326)])/32.
        self.PreFSize = 416 // 32

        # assert num_anchors==len(self.PreBoxSize),print("num_anchors:%d not equal num of PreBoxSize"%(num_anchors))


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
