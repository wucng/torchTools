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

def weights_init_fpn(model):
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

def weights_init_rpn(model):
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
    def __init__(self,cfg):
        super(BackBoneNet, self).__init__()

        model_name = cfg["network"]["backbone"]['model_name']
        pretrained = cfg["network"]["backbone"]['pretrained']
        freeze_at = cfg["network"]["backbone"]['freeze_at']
        self.out_features = cfg["network"]["backbone"]['out_features']

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

        layer = {"res1":layer0,"res2":_model.layer1,"res3":_model.layer2,"res4":_model.layer3,"res5":_model.layer4}
        while True:
            if freeze_at in layer: # 冻结某层参数
                layer[freeze_at]=freeze(layer[freeze_at])
                freeze_at =freeze_at[:-1] + str(int(freeze_at[-1])-1)
            else:
                break

        self.layer = nn.ModuleList(
            [nn.ModuleDict({"res1": layer["res1"]}),
             nn.ModuleDict({"res2": layer["res2"]}),
             nn.ModuleDict({"res3": layer["res3"]}),
             nn.ModuleDict({"res4": layer["res4"]}),
             nn.ModuleDict({"res5": layer["res5"]})
             ]
        )


    def forward(self, x):
        out = {}
        for item in self.layer:
            for k,net in item.items():
                x = net(x)
                if k in self.out_features:
                    out[k] = x
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
    def __init__(self,cfg,backbone_size=2048):
        super(FPNNet, self).__init__()
        usize = cfg["network"]["FPN"]["usize"]
        self.backbone_out_features = cfg["network"]["backbone"]['out_features'][::-1] # 反正下
        self.name_features = cfg["network"]["FPN"]['name_features'][::-1] # 反正下
        num_features = len(self.backbone_out_features)
        self.out_features = cfg["network"]["FPN"]['out_features']
        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, usize, 1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            p = nn.Sequential(
                nn.Conv2d(usize, usize, 3, stride=1, padding=1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            self.net.append(nn.ModuleList([m,p]))

    def forward(self, x_dict):
        out = {}
        out_m = None
        for i,key in enumerate(self.backbone_out_features):
            m,p=self.net[i]
            m_x = m(x_dict[key])
            if i>0:
                m_x += F.interpolate(out_m,scale_factor=(2,2),mode="nearest")
            p_x = p(m_x)
            out_m = m_x

            if self.name_features[i] in self.out_features:
                out[self.name_features[i]] = p_x

        return out

class FPNNetCH(nn.Module):
    def __init__(self,cfg,backbone_size=2048):
        super(FPNNetCH, self).__init__()
        usize = cfg["network"]["FPN"]["usize"]
        self.backbone_out_features = cfg["network"]["backbone"]['out_features'][::-1]  # 反正下
        self.name_features = cfg["network"]["FPN"]['name_features'][::-1]  # 反正下
        num_features = len(self.backbone_out_features)
        self.out_features = cfg["network"]["FPN"]['out_features']

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, backbone_size//2**i, 1),
                nn.BatchNorm2d(backbone_size//2**i),
                nn.LeakyReLU(0.2)
            )

            p = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, usize, 3, stride=1, padding=1),
                nn.BatchNorm2d(usize),
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

            self.net.append(nn.ModuleList([m,p,upsample]))

    def forward(self, x_dict):
        out = {}
        out_m = None
        for i,key in enumerate(self.backbone_out_features):
            m,p,upsample=self.net[i]
            m_x = m(x_dict[key])
            if i>0:
                # m_x += F.interpolate(out_m,scale_factor=(2,2),mode="nearest")
                m_x += upsample(out_m)
            p_x = p(m_x)
            out_m = m_x

            if self.name_features[i] in self.out_features:
                out[self.name_features[i]] = p_x

        return out

class FPNNetLarger(nn.Module):
    def __init__(self,cfg,backbone_size=2048):
        super(FPNNetLarger, self).__init__()
        usize = cfg["network"]["FPN"]["usize"]
        self.backbone_out_features = cfg["network"]["backbone"]['out_features'][::-1]  # 反正下
        self.name_features = cfg["network"]["FPN"]['name_features'][::-1]  # 反正下
        num_features = len(self.backbone_out_features)
        self.out_features = cfg["network"]["FPN"]['out_features']

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, backbone_size//2**i, 1),
                nn.BatchNorm2d(backbone_size//2**i),
                nn.LeakyReLU(0.2)
            )

            p = nn.Sequential(
                nn.Conv2d(backbone_size // 2 ** i, usize, 3, stride=1, padding=1),
                nn.BatchNorm2d(usize),
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

            self.net.append(nn.ModuleList([m,p,upsample]))

    def forward(self, x_dict):
        out = {}
        out_m = None
        for i, key in enumerate(self.backbone_out_features):
            m, p, upsample = self.net[i]
            m_x = m(x_dict[key])
            if i > 0:
                # m_x += F.interpolate(out_m,scale_factor=(2,2),mode="nearest")
                m_x += upsample(out_m)
            p_x = p(m_x)
            out_m = m_x

            if self.name_features[i] in self.out_features:
                out[self.name_features[i]] = p_x

        return out

class FPNNetSmall(nn.Module):
    def __init__(self,cfg,backbone_size=2048):
        super(FPNNetSmall, self).__init__()
        usize = cfg["network"]["FPN"]["usize"]
        self.backbone_out_features = cfg["network"]["backbone"]['out_features'][::-1]  # 反正下
        self.name_features = cfg["network"]["FPN"]['name_features'][::-1]  # 反正下
        num_features = len(self.backbone_out_features)
        self.out_features = cfg["network"]["FPN"]['out_features']

        self.net = nn.ModuleList()
        for i in range(num_features):
            m = nn.Sequential(
                nn.Conv2d(backbone_size//2**i, usize, 1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            if i > 0:
                upsample = nn.Sequential(
                    nn.ConvTranspose2d(usize,usize,3,2,1,1),
                    nn.BatchNorm2d(usize),
                    nn.LeakyReLU(0.2),
                )
            else:
                upsample = None

            self.net.append(nn.ModuleList([m,upsample]))

    def forward(self, x_dict):
        out = {}
        out_m = None
        for i, key in enumerate(self.backbone_out_features):
            m,upsample=self.net[i]
            m_x = m(x_dict[key])
            if i>0:
                # m_x += F.interpolate(out_m[-1],scale_factor=(2,2),mode="nearest")
                m_x += upsample(out_m)
            out_m = m_x

            if self.name_features[i] in self.out_features:
                out[self.name_features[i]] = m_x

        return out

class XNet(nn.Module):
    """
    # https://github.com/arashwan/matrixnet
    # MatrixNets（xNets）在FPN的基础上增加对feature做width和height的下采样
    # https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/103998265
    """
    def __init__(self,cfg,backbone_size=2048):
        super(XNet, self).__init__()
        usize = cfg["network"]["FPN"]["usize"]
        self.out_features = cfg["network"]["FPN"]['out_features']
        num_features = len(self.out_features)

        self.fpn = FPNNet(cfg,backbone_size)

        self.net = nn.ModuleList()
        for i in range(num_features):
            downsample_w1 = nn.Sequential(
                nn.Conv2d(usize,usize,3,(1,2),1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            downsample_w2 = nn.Sequential(
                nn.Conv2d(usize, usize, 3, (1, 2), 1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            downsample_h1 = nn.Sequential(
                nn.Conv2d(usize, usize, 3, (2, 1), 1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            downsample_h2 = nn.Sequential(
                nn.Conv2d(usize, usize, 3, (2, 1), 1),
                nn.BatchNorm2d(usize),
                nn.LeakyReLU(0.2)
            )

            self.net.append(nn.ModuleList([downsample_w1,downsample_w2,downsample_h1,downsample_h2]))

        self.num_features = (num_features-1)*2+1

    def forward(self, x_dict):
        new_out = {}
        outs = self.fpn(x_dict)
        for i,key in enumerate(self.out_features):
            dw1, dw2, dh1, dh2 = self.net[i]
            if key in ["p2","p3"]:
                new_out[key+"_w"]=dw2(dw1(outs[key]))
                new_out[key+"_h"]=dh2(dh1(outs[key]))
            elif key in ["p4"]:
                new_out[key + "_w"] = dw1(outs[key])
                new_out[key + "_h"] = dh1(outs[key])
            else:
                new_out[key] = outs[key]

        return new_out


class RPN(nn.Module):
    def __init__(self,cfg):
        super(RPN,self).__init__()
        in_channels = cfg["network"]["RPN"]["in_channels"]
        num_boxes = cfg["network"]["RPN"]["num_boxes"]
        num_classes = cfg["network"]["RPN"]["num_classes"]
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits # +1 包括背景
        self.pred_cls = nn.Conv2d(in_channels, num_boxes*(num_classes+1), kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.pred_box = nn.Conv2d(
            in_channels, num_boxes * 4, kernel_size=1, stride=1
        )

        # for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
        #     nn.init.normal_(l.weight, std=0.01)
        #     nn.init.constant_(l.bias, 0)

    def forward(self, features):
        pred_cls = {}
        pred_box = {}
        for k,x in features.items():
            t = F.relu(self.conv(x))
            bs = t.size(0)
            pred_cls[k]=self.pred_cls(t).permute(0, 2, 3, 1).contiguous().view(bs,-1,self.num_classes+1)
            pred_box[k]=self.pred_box(t).permute(0, 2, 3, 1).contiguous().view(bs,-1,4)

        # 排序
        keys = sorted(features.keys())
        last_pred_cls=pred_cls[keys[0]]
        last_pred_box=pred_box[keys[0]]
        if len(keys)>1:
            for key in keys[1:]:
                last_pred_cls = torch.cat((last_pred_cls,pred_cls[key]),1)
                last_pred_box = torch.cat((last_pred_box,pred_box[key]),1)

        return last_pred_cls, last_pred_box

if __name__=="__main__":
    import sys
    sys.path.append("..")
    from config.defaults import cfg
    backbone = BackBoneNet(cfg)
    # backbone.apply(weights_init)
    x = torch.rand([1,3,224,224])
    features = backbone(x)
    fpn = FPNNet(cfg,backbone.backbone_size)
    features = fpn(features)
    rpn = RPN(cfg)
    out = rpn(features)
    print()
    # torch.save(backbone.state_dict(),"model.pt")