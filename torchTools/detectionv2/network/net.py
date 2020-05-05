"""
可以用于：yolo,ssd,fasterrcnn
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

fpn_dict = {"FPNNet":FPNNet,"FPNNetCH":FPNNetCH,"FPNNetLarger":FPNNetLarger,"FPNNetSmall":FPNNetSmall,"XNet":XNet}

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.backbone = BackBoneNet(cfg)
        self.use_FPN = cfg["network"]["FPN"]["use_FPN"]
        fpn_name = cfg["network"]["FPN"]["name"]

        if self.use_FPN:
            self.fpn = fpn_dict[fpn_name](cfg,self.backbone.backbone_size)

        self.rpn = RPN(cfg)

    def forward(self,x):
        features = self.backbone(x)
        if self.use_FPN:
            features = self.fpn(features)
        out = self.rpn(features)
        return out


class FasterRCNN(nn.Module):
    def __init__(self, cfg):
        super(FasterRCNN, self).__init__()
        self.backbone = BackBoneNet(cfg)
        self.use_FPN = cfg["network"]["FPN"]["use_FPN"]
        fpn_name = cfg["network"]["FPN"]["name"]

        self.align_features = cfg["network"]["RCNN"]["align_features"]

        if self.use_FPN:
            self.fpn = fpn_dict[fpn_name](cfg,self.backbone.backbone_size)

        self.rpn = RPN(cfg)
        self.rcnn = RCNN(cfg)

    def forward(self,x):
        features = self.backbone(x)
        if self.use_FPN:
            features = self.fpn(features)
        out = self.rpn(features)
        return out,features[self.align_features]

    def doRCNN(self,features):
        out=[]
        for feature in features:
            out.append(self.rcnn(feature))
        return out


if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from config.config import *
    cfg = get_cfg()
    cfg2={
        "network":{
            "backbone":{
                "out_features":["res5"]
            },
            "FPN":{
                "use_FPN":False
            },
            "RPN":{
                "in_channels":512
            }
        }
    }
    # cfg["network"]["backbone"]["out_features"] = ["res5"]
    # cfg["network"]["FPN"]["use_FPN"] = False
    # cfg["network"]["RPN"]["in_channels"] = 512
    cfg = merge_from_cfg(cfg,cfg2)
    net = Network(cfg)
    x = torch.rand([1,3,224,224])
    out = net(x)
    print()
    # torch.save(net.state_dict(), "./model.pt")