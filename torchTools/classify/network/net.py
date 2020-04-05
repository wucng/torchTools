"""
torchvision 内置的神经网络模块(torchvision.models)
"""
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict

__all__ = ['Resnet', 'Mnasnet', 'Densenet',
           'Alexnet','VGGnet','Squeezenet',
           'Mobilenet','ShuffleNetV2']


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return torch.flatten(x,1)
        # return x.view(x.size(0), -1)

class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()

    def params(self):
        base_params = list(
            map(id, self.backbone.parameters())
        )
        logits_params = filter(lambda p: id(p) not in base_params, self.parameters())

        params = [
            {"params": logits_params, "lr": 1e-3},
            {"params": self.backbone.parameters(), "lr": 2e-4},
        ]

        return params

# model resnet
class Resnet(BaseNet):
    def __init__(self, num_classes, model_name="resnet101", pretrained=False, droprate=0.0):
        super(Resnet, self).__init__()
        self.pretrained = pretrained

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

        backbone_size = model_dict[model_name]

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

        self._conv1 = nn.Sequential(
            nn.Dropout(droprate, inplace=True),
            # nn.Conv2d(backbone_size, backbone_size, 3, 1,padding=1),
            # nn.BatchNorm2d(backbone_size),
            # nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(backbone_size, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)

        return x

# model mnasnet
class Mnasnet(BaseNet):
    def __init__(self, num_classes, model_name="mnasnet0_5", pretrained=False, dropout=0.0):
        super(Mnasnet, self).__init__()
        self.pretrained = pretrained

        model_dict = {'mnasnet0_5': 1280,
                      'mnasnet0_75': 1280,
                      'mnasnet1_0': 1280,
                      'mnasnet1_3': 1280}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        backbone_size = model_dict[model_name]

        _model = torchvision.models.mnasnet.__dict__[model_name](pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict([
            ('layers', _model.layers)
        ]))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            # nn.Conv2d(backbone_size, backbone_size, 3, 1,padding=1),
            # nn.BatchNorm2d(backbone_size),
            # nn.ReLU(),
            nn.Linear(backbone_size, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        # x = x.mean([2, 3])
        return self.classifier(x)

# model densenet
class Densenet(BaseNet):
    def __init__(self, num_classes, model_name="densenet121",pretrained=False, droprate=0.0):
        super(Densenet, self).__init__()
        self.pretrained = pretrained

        model_dict = {'densenet121': 1024,
                      'densenet169': 1664,
                      'densenet201': 1920,
                      'densenet161': 2208}

        assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

        backbone_size = model_dict[model_name]

        _model = torchvision.models.densenet.__dict__[model_name](pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict([
            ('features', _model.features)
        ]))

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=droprate, inplace=True),
            nn.Linear(backbone_size, num_classes))

    def forward(self, x):
        features = self.backbone(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

class Alexnet(BaseNet):
    def __init__(self, num_classes, pretrained=False, droprate=0.0):
        super(Alexnet, self).__init__()
        self.pretrained = pretrained

        # _model = torchvision.models.alexnet.__dict__[model_name](pretrained=pretrained)
        _model = torchvision.models.alexnet(pretrained=pretrained)

        self.backbone = nn.Sequential(OrderedDict([
            ('features', _model.features)
        ]))

        self._conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            Flatten(),
            nn.Dropout(droprate, inplace=True),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(droprate, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)

        return x

class VGGnet(BaseNet):
    def __init__(self, num_classes, model_name="vgg16", pretrained=False, droprate=0.0):
        super(VGGnet, self).__init__()
        self.pretrained = pretrained

        model_dict ={'vgg11':512,
                     'vgg13':512,
                     'vgg16':2048,
                     'vgg19':2048,
                      'vgg11_bn':2048,
                     'vgg13_bn':2048,
                     'vgg16_bn':2048,
                     'vgg19_bn':2048
                     }

        assert model_name in model_dict,"%s must be in %s"%(model_name,model_dict.keys())

        # backbone_size = model_dict[model_name]

        _model = torchvision.models.vgg.__dict__[model_name](pretrained=pretrained)

        self.backbone = nn.Sequential(OrderedDict([
            ('features', _model.features)
        ]))

        self._conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=droprate, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=droprate, inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)

        return x

class Squeezenet(BaseNet):
    def __init__(self, num_classes, model_name="squeezenet1_0", pretrained=False, droprate=0.0):
        super(Squeezenet, self).__init__()
        self.pretrained = pretrained

        model_dict ={'squeezenet1_0':512,
                     'squeezenet1_1':512,
                     }

        assert model_name in model_dict,"%s must be in %s"%(model_name,model_dict.keys())

        # backbone_size = model_dict[model_name]

        _model = torchvision.models.squeezenet.__dict__[model_name](pretrained=pretrained)

        self.backbone = nn.Sequential(OrderedDict([
            ('features', _model.features)
        ]))

        self._conv1 = nn.Sequential(
            nn.Dropout(p=droprate, inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)

        return x

class Mobilenet(BaseNet):
    def __init__(self, num_classes, model_name="mobilenet_v2", pretrained=False, droprate=0.0):
        super(Mobilenet, self).__init__()
        self.pretrained = pretrained

        model_dict ={'mobilenet_v2':62720,
                     }

        assert model_name in model_dict,"%s must be in %s"%(model_name,model_dict.keys())

        backbone_size = model_dict[model_name]

        _model = torchvision.models.mobilenet.__dict__[model_name](pretrained=pretrained)

        self.backbone = nn.Sequential(OrderedDict([
            ('features', _model.features)
        ]))

        self._conv1 = nn.Sequential(
            nn.Dropout(p=droprate, inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(backbone_size, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)

        return x

class ShuffleNetV2(BaseNet):
    def __init__(self, num_classes, model_name="shufflenetv2_x0.5", pretrained=False, droprate=0.0):
        super(ShuffleNetV2, self).__init__()
        self.pretrained = pretrained

        model_dict ={
            'shufflenetv2_x0.5':1024,
            'shufflenetv2_x1.0':1024,
            'shufflenetv2_x1.5':1024,
            'shufflenetv2_x2.0':2048,
             }

        assert model_name in model_dict,"%s must be in %s"%(model_name,model_dict.keys())

        backbone_size = model_dict[model_name]

        # _model = torchvision.models.shufflenetv2.__dict__[model_name](pretrained=pretrained)
        if model_name=="shufflenetv2_x0.5":
            _model = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
        elif model_name=="shufflenetv2_x1.0":
            _model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
        elif model_name=="shufflenetv2_x1.5":
            _model = torchvision.models.shufflenet_v2_x1_5(pretrained=pretrained)
        else:
            _model = torchvision.models.shufflenet_v2_x2_0(pretrained=pretrained)

        self.backbone = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('maxpool', _model.maxpool),
            ('stage2', _model.stage2),
            ('stage3', _model.stage3),
            ('stage4', _model.stage4),
            ('conv5', _model.conv5),
        ]))

        self._conv1 = nn.Sequential(
            nn.Dropout(p=droprate, inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(),
            nn.Linear(backbone_size, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self._conv1(x)

        return x

if __name__=="__main__":
    model = Resnet(10,'resnet18')
    x = torch.rand([1,3,224,224])

    print(model(x).shape)