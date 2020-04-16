"""
torchvision 内置的神经网络模块(torchvision.models)
"""
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict
import numpy as np

__all__ = ['Resnet', 'Mnasnet', 'Densenet',
           'Alexnet','VGGnet','Squeezenet',
           'Mobilenet','ShuffleNetV2']

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# model resnet
class BackBoneNet(nn.Module):
    def __init__(self,model_name="resnet101", pretrained=False,dropRate=0.5):
        super(BackBoneNet, self).__init__()
        self.pretrained = pretrained
        # self.dropRate = dropRate

        model_dict ={'resnet18':512,
                     'resnet34':512,
                     'resnet50':2048,
                     'resnet101':2048,
                      'resnet152':2048,
                     'resnext50_32x4d':2048,
                     'resnext101_32x8d':2048,
                     'wide_resnet50_2':2048,
                     'wide_resnet101_2':2048}

        assert model_name in model_dict,"%s must be in %s"%(model_name,model_dict.keys())

        self.backbone_size = model_dict[model_name]

        _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)


        layer0 = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),
            ('relu1', _model.relu),
            ('maxpool1', _model.maxpool),
        ]))

        self.backbone = nn.ModuleList()
        self.backbone.append(layer0)
        self.backbone.append(nn.Sequential(_model.layer1,nn.Dropout(dropRate)))
        self.backbone.append(nn.Sequential(_model.layer2,nn.Dropout(dropRate)))
        self.backbone.append(nn.Sequential(_model.layer3,nn.Dropout(dropRate)))
        self.backbone.append(nn.Sequential(_model.layer4,nn.Dropout(dropRate)))

        self.num_features = len(self.backbone)-1

    def forward(self, x):
        out = []
        for i,net in enumerate(self.backbone):
            x = net(x)
            if i>0:
                out.append(x)
        return out

class FPNNet(nn.Module):
    def __init__(self,backbone_size=2048,num_features=4, usize=256):
        super(FPNNet, self).__init__()
        self.num_features = num_features

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, usize, 1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            upsample = nn.Sequential(
                nn.ConvTranspose2d(usize,usize,3,2,1,1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2),
            )

            tmp = nn.ModuleList()
            tmp.append(m)
            tmp.append(upsample)
            self.net.append(tmp)

    def forward(self, x_list):
        x_list = x_list[::-1] # 反转
        out = []
        for i in range(self.num_features):
            m,upsample=self.net[i]
            m_x = m(x_list[i])
            if i>0:
                # m_x += F.interpolate(out_m[-1],scale_factor=(2,2))
                m_x += upsample(out[-1])
            out.append(m_x)

        return out[::-1] # 反转


class XNet(nn.Module):
    """
    # https://github.com/arashwan/matrixnet
    # MatrixNets（xNets）在FPN的基础上增加对feature做width和height的下采样
    # https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/103998265
    """
    def __init__(self,backbone_size=2048,num_features=4, usize=256):
        super(XNet, self).__init__()
        self.num_features = num_features
        self.fpn = FPNNet(backbone_size, num_features, usize)

        self.net = nn.ModuleList()
        for i in range(self.num_features-1):
            downsample_w1 = nn.Sequential(
                nn.Conv2d(usize,usize,3,(1,2),1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            downsample_w2 = nn.Sequential(
                nn.Conv2d(usize, usize, 3, (1, 2), 1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            downsample_h1 = nn.Sequential(
                nn.Conv2d(usize, usize, 3, (2, 1), 1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            downsample_h2 = nn.Sequential(
                nn.Conv2d(usize, usize, 3, (2, 1), 1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            tmp = nn.ModuleList()
            tmp.append(downsample_w1)
            tmp.append(downsample_w2)
            tmp.append(downsample_h1)
            tmp.append(downsample_h2)
            self.net.append(tmp)

        self.num_features = (self.num_features-1)*2+1

    def forward(self, x_list):
        new_out = []
        outs = self.fpn(x_list)
        len_outs = len(outs)
        for i,out in enumerate(outs[:-1]):
            dw1,dw2,dh1,dh2= self.net[i]

            if i < len_outs-2:
                new_out.append(dw2(dw1(out)))
                new_out.append(dh2(dh1(out)))
            else:
                new_out.append(dw1(out))
                new_out.append(dh1(out))

        new_out.append(outs[-1])

        return new_out

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




if __name__ == "__main__":
    net = YOLOV1Net(model_name="resnet18", usize=256,num_features=4)
    # x = torch.rand([5,3,224,224])
    # print(net(x).shape)
    torch.save(net.state_dict(), "./model.pt")