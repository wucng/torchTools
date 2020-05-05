import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor,FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor,MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn
from collections import OrderedDict
# from torchvision.ops import misc as misc_nn_ops

# class FasterRCNN0(object):
def FasterRCNN0(num_classes=2,pretrained=False):
    """
    :param num_classes:包括background
    :return:
    """
    # super(FasterRCNN0, self).__init__()
    # self.num_classes = num_classes

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# 自定义
def FasterRCNN1(num_classes=2, model_name="resnet101", pretrained=False,usize = 256,use_FPN=False):
    # super(FasterRCNN1, self).__init__()
    model_dict = {'resnet18': 512,
                  'resnet34': 512,
                  'resnet50': 2048,
                  'resnet101': 2048,
                  'resnet152': 2048,
                  'resnext50_32x4d': 2048,
                  'resnext101_32x8d': 2048,
                  'wide_resnet50_2': 2048,
                  'wide_resnet101_2': 2048}

    assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

    backbone_size = model_dict[model_name]

    _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)

    # backbone = resnet.__dict__[model_name](
    #     pretrained=pretrained,
    #     norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    backbone = nn.Sequential(OrderedDict([
        ('conv1', _model.conv1),
        ('bn1', _model.bn1),
        ('relu1', _model.relu),
        ('maxpool1', _model.maxpool),

        ("layer1", _model.layer1),
        ("layer2", _model.layer2),
        ("layer3", _model.layer3),
        ("layer4", _model.layer4),
    ]))

    if use_FPN:
        # freeze layers (layer1)
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list = [
            backbone_size // 8,  # 64 layer1 输出特征维度
            backbone_size // 4,  # 128 layer2 输出特征维度
            backbone_size // 2,  # 256 layer3 输出特征维度
            backbone_size,       # 512 layer4 输出特征维度
        ]

        out_channels = usize  # 每个FPN层输出维度 (这个值不固定，也可以设置为64,512等)

        backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

        model = FasterRCNN(backbone, num_classes)
    else:
        backbone.out_channels = model_dict[model_name]  # 特征的输出维度
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                        output_size=7,
                                                        sampling_ratio=2)

        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)

    return model


def get_instance_segmentation_model(num_classes,pretrained=False,hidden_layer = 256):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# 2 - Modifying the model to add a different backbone
# 自定义
def get_instance_segmentation_model_cum(num_classes=2, model_name="resnet101", pretrained=False,usize = 256,use_FPN=False):
    # super(FasterRCNN1, self).__init__()
    model_dict = {'resnet18': 512,
                  'resnet34': 512,
                  'resnet50': 2048,
                  'resnet101': 2048,
                  'resnet152': 2048,
                  'resnext50_32x4d': 2048,
                  'resnext101_32x8d': 2048,
                  'wide_resnet50_2': 2048,
                  'wide_resnet101_2': 2048}

    assert model_name in model_dict, "%s must be in %s" % (model_name, model_dict.keys())

    backbone_size = model_dict[model_name]

    _model = torchvision.models.resnet.__dict__[model_name](pretrained=pretrained)

    # backbone = resnet.__dict__[model_name](
    #     pretrained=pretrained,
    #     norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    backbone = nn.Sequential(OrderedDict([
        ('conv1', _model.conv1),
        ('bn1', _model.bn1),
        ('relu1', _model.relu),
        ('maxpool1', _model.maxpool),

        ("layer1", _model.layer1),
        ("layer2", _model.layer2),
        ("layer3", _model.layer3),
        ("layer4", _model.layer4),
    ]))

    if use_FPN:
        # freeze layers (layer1)
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list = [
            backbone_size//8,      # 64 layer1 输出特征维度
            backbone_size//4,  # 128 layer2 输出特征维度
            backbone_size//2,  # 256 layer3 输出特征维度
            backbone_size,  # 512 layer4 输出特征维度
        ]

        out_channels = usize  # 每个FPN层输出维度 (这个值不固定，也可以设置为64,512等)

        backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

        # model = FasterRCNN(backbone, num_classes)
        model = MaskRCNN(backbone, num_classes)
    else:
        backbone.out_channels = model_dict[model_name]  # 特征的输出维度
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[str(0)],
                                                        output_size=7,
                                                        sampling_ratio=2)

        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[str(0)],
                                                             output_size=14, sampling_ratio=2)

        # model = FasterRCNN(backbone,
        #                    num_classes=num_classes,
        #                    rpn_anchor_generator=anchor_generator,
        #                    box_roi_pool=roi_pooler)

        model = MaskRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)

    return model