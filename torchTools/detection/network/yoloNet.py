"""
YOLO 系列
"""
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict
import numpy as np
try:
    from .baseNet import *
except:
    from baseNet import *

class YOLOV1Net(nn.Module):
    def __init__(self, num_classes=1, num_anchors=2, model_name="resnet101",num_features=None,
                pretrained=False, dropRate=0.5, usize=256):
        super(YOLOV1Net, self).__init__()
        self.backbone = BackBoneNet(model_name,pretrained,dropRate)
        self.num_features = self.backbone.num_features if num_features is None else num_features
        self.fpn = FPNNet(self.backbone.backbone_size,self.num_features,usize)
        # self.fpn = XNet(self.backbone.backbone_size,self.num_features,usize)
        self.num_features = self.fpn.num_features
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.net = nn.ModuleList()
        for i in range(self.num_features):
            convp = nn.Sequential(
                nn.Dropout(dropRate),
                # nn.Conv2d(usize, usize, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(usize),
                # nn.LeakyReLU(),
                nn.Conv2d(usize, num_anchors * (5 + num_classes), kernel_size=3, stride=1, padding=1),
                # 每个box对应一个类别
                # 每个anchor对应4个坐标
                nn.BatchNorm2d(num_anchors * (5 + num_classes)),
                nn.Sigmoid()
            )

            self.net.append(convp)

    def forward(self,x):
        x_list = self.backbone(x)
        p_x = self.fpn(x_list)
        out = []
        for i in range(self.num_features):
            p = self.net[i](p_x[i])
            p = p.permute(0, 2, 3, 1)  # (-1,7,7,30)

            out.append(p)

        return out # p1,p2,p3,p4

class YOLOV2Net(nn.Module):
    def __init__(self, num_classes=1, num_anchors=2, model_name="resnet101",num_features=None,
                pretrained=False, dropRate=0.5, usize=256):
        super(YOLOV2Net, self).__init__()
        self.backbone = BackBoneNet(model_name,pretrained,dropRate)
        self.num_features = self.backbone.num_features if num_features is None else num_features
        self.fpn = FPNNet(self.backbone.backbone_size,self.num_features,usize)
        # self.fpn = XNet(self.backbone.backbone_size,self.num_features,usize)
        self.num_features = self.fpn.num_features
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.net = nn.ModuleList()
        for i in range(self.num_features):
            convp = nn.Sequential(
                nn.Dropout(dropRate),
                # nn.Conv2d(usize, usize, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(usize),
                # nn.LeakyReLU(),
                nn.Conv2d(usize, num_anchors * (5 + num_classes), kernel_size=3, stride=1, padding=1),
                # 每个box对应一个类别
                # 每个anchor对应4个坐标
                nn.BatchNorm2d(num_anchors * (5 + num_classes)),
                # nn.Sigmoid()
            )

            self.net.append(convp)

    def forward(self,x):
        x_list = self.backbone(x)
        p_x = self.fpn(x_list)
        out = []
        for i in range(self.num_features):
            p = self.net[i](p_x[i])
            p = p.permute(0, 2, 3, 1)  # (-1,7,7,30)
            # -----------------------
            bs,fh,fw,c = p.shape
            p = p.contiguous().view(bs,fh,fw, self.num_anchors,5 + self.num_classes)
            p[...,4:] = torch.sigmoid(p[...,4:])
            p[...,:2] = torch.sigmoid(p[...,:2])
            p = p.contiguous().view(bs, fh, fw, c)
            # -----------------------
            out.append(p)

        return out # p1,p2,p3,p4


if __name__ == "__main__":
    net = YOLOV1Net(model_name="resnet18", usize=256,num_features=4)
    x = torch.rand([5,3,224,224])
    print(net(x)[-1].shape)
    # torch.save(net.state_dict(), "./model.pt")