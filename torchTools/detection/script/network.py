"""
backbone使用resnet
"""
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
import os
from collections import OrderedDict
import math,json

# model
class YOLOV1Net(nn.Module):
    def __init__(self, num_classes, model_name="resnet101", backbone_size=2048, pretrained=False, droprate=0.0,
                 device="cpu"):
        super(YOLOV1Net, self).__init__()
        # self.pretrained = pretrained
        self.num_anchors = 2
        self.num_classes = num_classes

        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),
            ('relu1', _model.relu),
            ('maxpool1', _model.maxpool),

            ("layer1", _model.layer1),
            ("layer2", _model.layer2),
            ("layer3", _model.layer3),
            ("layer4", _model.layer4),
        ]))

        self.conv = nn.Sequential(
            nn.Dropout(droprate),
            nn.Conv2d(backbone_size, backbone_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(backbone_size),
            nn.LeakyReLU()
        )

        # 原始yolov1方式
        # self.pred = nn.Sequential(
        #     nn.Conv2d(backbone_size, num_anchors * 5 + num_classes, kernel_size=3, stride=1, padding=1),  # 每个anchor对应4个坐标
        #     nn.BatchNorm2d(num_anchors * 5 + num_classes),
        #     nn.Sigmoid()
        # )

        self.pred = nn.Sequential(
            nn.Conv2d(backbone_size, self.num_anchors * (5 + num_classes), kernel_size=3, stride=1, padding=1), # 每个box对应一个类别
            # 每个anchor对应4个坐标
            nn.BatchNorm2d(self.num_anchors * (5 + num_classes)),
            nn.Sigmoid()
        )

        # for l in self.children():
        #     torch.nn.init.normal_(l.weight, std=0.01)
        #     torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.pred(x)
        x = x.permute(0, 2, 3, 1)  # (-1,7,7,30)
        # x = x.view(-1,self.num_anchors * (5 + self.num_classes))
        # x = x.reshape((-1,self.num_anchors * (5 + self.num_classes)))
        # x = x.reshape((-1,5 + self.num_classes))
        return x


if __name__=="__main__":
    net = YOLOV1(1,"resnet18",512)
    x = torch.rand([5,3,224,224])
    print(net(x).shape)