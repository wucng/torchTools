try:
    from . import transforms as T
except:
    import transforms as T
import torch
import random
import numpy as np
from torch.nn import functional as F
import torchvision
try:
    from . import bboxAug
except:
    import bboxAug

def get_transform(train,multi_scale=False,deaful_size=416):
    transforms = []

    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(Augment())

    # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(Resize(min_size=800,training=multi_scale))
    transforms.append(Resize_fixed(img_size=deaful_size,training=multi_scale))

    # transforms.append(T.ToTensor()) # to 0~1
    transforms.append(ToTensor())  # to 0~1

    return T.Compose(transforms)


class ToTensor(object):
    def __init__(self, max_objects=50, is_debug=False,do_normalize=False,image_mean=None,image_std=None):
        self.max_objects = max_objects
        self.is_debug = is_debug

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.do_normalize=do_normalize
        self.image_mean=image_mean
        self.image_std=image_std

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def __call__(self, image, target):
        boxes,labels = target['boxes'], target['labels']

        if self.is_debug == False:
            # image = torchvision.transforms.functional.to_tensor(image)
            image=image/255.0

            # box归一化到输入图像(box/[img_w,img_h])
            img_h,img_w=image.size()[-2:]
            # tmp=torch.as_tensor([img_w,img_h,img_w,img_h],dtype=torch.float32,device=image.device).unsqueeze(0)
            # boxes=boxes/tmp # boxes 格式 (x1,y1,x2,y2)

            boxes[...,[0,2]]=boxes[...,[0,2]]/img_w
            boxes[...,[1,3]]=boxes[...,[1,3]]/img_h


        if self.do_normalize:
            image=self.normalize(image)

        filled_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32,device=boxes.device)
        filled_labels = torch.ones((self.max_objects, ), dtype=torch.int64,device=labels.device)*(-1)

        filled_boxes[range(len(boxes))[:self.max_objects]] = boxes[:self.max_objects]
        filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]

        target={"boxes":filled_boxes,"labels":filled_labels}

        return image,target


class Resize(object):
    def __init__(self,min_size=[800,864,928,992],max_size=1333,training=True):
        if not isinstance(min_size,(tuple,list)):
            min_size=[min_size]

        self.min_size=min_size
        self.max_size=max_size
        self.training=training

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = F.interpolate(
            image[None].float(), scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]
        image = image.clamp(min=0., max=255.)

        if target is None:
            target={}
            target["scale_factor"] = torch.as_tensor(scale_factor, device=image.device)
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def __call__(self, image, target):
        # normalizer
        # image = self.normalize(image)
        image, target = self.resize(image, target)

        return image, target

class Resize_fixed(object):
    def __init__(self,img_size=416,training=True,multi_scale=True): # 416
        # if not isinstance(img_size,(tuple,list)):
        #     img_size=[img_size]

        self.img_size=img_size
        self.training=training

        if multi_scale:
            """
            s = img_size / 32
            nb=10
            self.multi_scale = ((np.linspace(0.5, 2, nb) * s).round().astype(np.int) * 32)
            """
            # self.multi_scale=[320,352,384,416,448,480,512,544,576,608]
            self.multi_scale=[384,416,448,480,512,544,576,608,640,672]

    def pad_img(self,A,box=None, mode='constant', value=128.): # reflect
        h, w = A.size()[-2:]
        if h >= w:
            diff = h - w
            pad_list = [diff // 2, diff - diff // 2, 0, 0]
            if box is not None:
                box=[[b[0]+diff // 2,b[1],b[2]+diff // 2,b[3]] for b in box]
                box=torch.as_tensor(box)
        else:
            diff = w - h
            pad_list = [0, 0, diff // 2, diff - diff // 2]
            if box is not None:
                box = [[b[0], b[1]+diff // 2, b[2], b[3]+diff // 2] for b in box]
                box = torch.as_tensor(box)

        A_pad = F.pad(A, pad_list, mode=mode, value=value)

        return A_pad,box,h,w

    def resize(self, image, target):
        """先按最长边填充成正方形，在resize"""

        if self.training:
            size = random.choice(self.multi_scale)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.img_size

        if target is None:
            bbox=None
        else:
            bbox = target["boxes"]
        # pad
        image,bbox,h,w=self.pad_img(image,box=bbox)

        # resize
        image = F.interpolate(
            image[None].float(),size=(size,size), mode='bicubic', align_corners=False)[0] # bicubic bilinear
        image=image.clamp(min=0.,max=255.)


        if target is None:
            target={}
            target["scale_factor"] = torch.as_tensor([h,w,size], device=image.device)
            return image, target

        bbox = resize_boxes(bbox, (h,h) if h>w else (w,w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def resize_2(self,image, target):
        """先按比例resize，再填充"""
        if self.training:
            size = random.choice(self.multi_scale)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.img_size

        if target is None:
            bbox=None
        else:
            bbox = target["boxes"]

        # 按比例resize
        img_h, img_w = image.size()[-2:]
        w, h = size,size
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))

        if new_w>=new_h:
            new_w=max(new_w,size)
        else:
            new_h = max(new_h, size)

        # resize
        image = F.interpolate(
            image[None].float(), size=(new_h, new_w), mode='bicubic', align_corners=False)[0]  # bicubic bilinear
        image = image.clamp(min=0., max=255.)

        bbox = resize_boxes(bbox, (img_h, img_w), image.shape[-2:])
        # bbox[:,[0,2]]*= new_w/img_w
        # bbox[:,[1,3]]*= new_h/img_h

        # pad
        image, bbox, _, _ = self.pad_img(image, box=bbox)

        target["boxes"] = bbox

        return image, target

    def __call__(self, image, target):
        # 转换类型
        if type(image)!=torch.Tensor:
            image=np.transpose(image, [2, 0,1])
            image=torch.as_tensor(image)
            target["boxes"] = torch.as_tensor(target["boxes"])
            target["labels"] = torch.as_tensor(target["labels"])

        image, target = self.resize(image, target)
        # image, target = self.resize_2(image, target)

        return image, target

class Augment(object):
    def __call__(self, image, target):
        image, target=bboxAug.simple_agu(image,target,bboxAug.run_seq2(),
                                         ) # bboxAug.run_seq2() np.random.randint(1,1e5,1)

        return image,target


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    return resized_data

def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)