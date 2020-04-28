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
from fvcore.nn.weight_init import c2_xavier_fill,c2_msra_fill

# __all__ = ['Resnet', 'Mnasnet', 'Densenet',
#            'Alexnet','VGGnet','Squeezenet',
#            'Mobilenet','ShuffleNetV2']

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)

def weights_init(model):
    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # if m.bias is not None:
            #     nn.init.zeros_(m.bias)
            # c2_msra_fill(m)
            c2_xavier_fill(m)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def weights_init2(model):
    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# model resnet
class BackBoneNet(nn.Module):
    def __init__(self,model_name="resnet101", pretrained=False,dropRate=0.5,freeze_at:int=0):
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

        layer = {0:layer0,1:_model.layer1,2:_model.layer2,3:_model.layer3,4:_model.layer4}
        while True:
            if freeze_at in layer: # 冻结某层参数
                layer[freeze_at]=freeze(layer[freeze_at])
                freeze_at -= 1
            else:
                break

        self.backbone = nn.ModuleList()
        # self.backbone.append(layer[0])
        # self.backbone.append(nn.Sequential(layer[1],nn.Dropout(dropRate)))
        # self.backbone.append(nn.Sequential(layer[2],nn.Dropout(dropRate)))
        # self.backbone.append(nn.Sequential(layer[3],nn.Dropout(dropRate)))
        # self.backbone.append(nn.Sequential(layer[4],nn.Dropout(dropRate)))
        for i in range(len(layer)):
            self.backbone.append(layer[i])

        self.num_features = len(self.backbone)-1

    def forward(self, x):
        out = []
        for i,net in enumerate(self.backbone):
            x = net(x)
            if i>0:
                out.append(x)
        return out

def freeze(model:nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    convert_frozen_batchnorm(model)
    return model

def convert_frozen_batchnorm(module:nn.Module):
    """
            Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

            Args:
                module (torch.nn.Module):

            Returns:
                If module is BatchNorm/SyncBatchNorm, returns a new module.
                Otherwise, in-place convert module and return it.

            Similar to convert_sync_batchnorm in
            https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
            """
    bn_module = nn.modules.batchnorm
    bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
    res = module
    if isinstance(module, bn_module):
        # res = module.num_features
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for name, child in module.named_children():
            new_child = convert_frozen_batchnorm(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


class FPNNet(nn.Module):
    def __init__(self,backbone_size=2048,num_features=4, usize=256):
        super(FPNNet, self).__init__()
        self.num_features = num_features

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, usize, 1),
                # nn.BatchNorm2d(usize),
                # nn.ReLU()
                # nn.LeakyReLU(0.2)
            )

            p = nn.Sequential(
                nn.Conv2d(usize, usize, 3, stride=1, padding=1),
                # nn.BatchNorm2d(usize),
                # nn.ReLU()
                # nn.LeakyReLU(0.2)
            )

            # if i>0:
            #     upsample = nn.Sequential(
            #         nn.ConvTranspose2d(backbone_size//2**(i-1),usize,3,2,1,1),
            #         # nn.BatchNorm2d(usize),
            #         # nn.LeakyReLU(0.2),
            #     )
            # else:
            upsample = None

            tmp = nn.ModuleList()
            tmp.append(m)
            tmp.append(p)
            tmp.append(upsample)
            self.net.append(tmp)

    def forward(self, x_list):
        x_list = x_list[::-1] # 反转
        out = []
        out_m = []
        for i in range(self.num_features):
            m,p,upsample=self.net[i]
            m_x = m(x_list[i])
            if i>0:
                m_x += F.interpolate(out_m[-1],scale_factor=(2,2),mode="nearest")
                # m_x += upsample(out_m[-1])
            p_x = p(m_x)

            out.append(p_x)
            out_m.append(m_x)

        return out[::-1] # 反转


class FPNNetCH(nn.Module):
    def __init__(self,backbone_size=2048,num_features=4, usize=256):
        super(FPNNetCH, self).__init__()
        self.num_features = num_features

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, backbone_size//2**i, 1),
                nn.BatchNorm2d(backbone_size//2**i),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            p = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, usize, 3, stride=1, padding=1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            if i>0:
                upsample = nn.Sequential(
                    nn.ConvTranspose2d(backbone_size//2**(i-1),backbone_size//2**i,3,2,1,1),
                    nn.BatchNorm2d(backbone_size//2**i),
                    nn.LeakyReLU(0.2),
                )
            else:
                upsample = None

            tmp = nn.ModuleList()
            tmp.append(m)
            tmp.append(p)
            tmp.append(upsample)
            self.net.append(tmp)

    def forward(self, x_list):
        x_list = x_list[::-1] # 反转
        out = []
        out_m = []
        for i in range(self.num_features):
            m,p,upsample=self.net[i]
            m_x = m(x_list[i])
            if i>0:
                # m_x += F.interpolate(out_m[-1],scale_factor=(2,2))
                m_x += upsample(out_m[-1])
            p_x = p(m_x)

            out.append(p_x)
            out_m.append(m_x)

        return out[::-1] # 反转

class FPNNetLarger(nn.Module):
    def __init__(self,backbone_size=2048,num_features=4, usize=256):
        super(FPNNetLarger, self).__init__()
        self.num_features = num_features

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, backbone_size//2**i, 1),
                nn.BatchNorm2d(backbone_size//2**i),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )

            if i>0:
                upsample = nn.Sequential(
                    nn.ConvTranspose2d(backbone_size//2**(i-1),backbone_size//2**i,3,2,1,1),
                    nn.BatchNorm2d(backbone_size//2**i),
                    nn.LeakyReLU(0.2)
                )
            else:
                upsample = None

            p = nn.Sequential(
                nn.Conv2d(backbone_size // 2 ** i, usize, 3, stride=1, padding=1),
                nn.BatchNorm2d(usize),
                # nn.ReLU()
                nn.LeakyReLU(0.2)
            )


            tmp = nn.ModuleList()
            tmp.append(m)
            tmp.append(upsample)
            tmp.append(p)
            self.net.append(tmp)

    def forward(self, x_list):
        x_list = x_list[::-1] # 反转
        out = []
        out_x = []
        for i in range(self.num_features):
            m,upsample,p=self.net[i]
            m_x = m(x_list[i])
            if i>0:
                # m_x += F.interpolate(out_m[-1],scale_factor=(2,2))
                m_x += upsample(out[-1])
            out.append(m_x)
            out_x.append(p(m_x))

        # return out[::-1] # 反转
        return out_x[::-1] # 反转

class FPNNetSmall(nn.Module):
    def __init__(self,backbone_size=2048,num_features=4, usize=256):
        super(FPNNetSmall, self).__init__()
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

if __name__=="__main__":
    backbone = BackBoneNet("resnet18")
    backbone.apply(weights_init)