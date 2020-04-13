"""
每经过10批训练（10 batches）就会随机选择新的图片尺寸，尺度定义为32的倍数，（ 320,352，…，608 ）
"""

from __future__ import print_function
import xml.etree.ElementTree as ET
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torch
import json
import cv2
import os
from PIL import Image
import sys
import random
try:
    from .do_transforms import get_transform
except:
    from do_transforms import get_transform
# from torch.utils.data import DataLoader,Dataset

class Generator(object):

    def __init__(self,DS,batch_size,shuffle=True,seed=1):
        self.DS=DS
        self.batch_size=batch_size
        self.index=0
        if shuffle: self.DS._shuffle(seed)

        self.num_example=len(self.DS)

    # def __iter__(self):
    #     while True:
    #         strat = self.index * self.batch_size
    #         end = (self.index + 1) * self.batch_size
    #         batch_x, batch_y=self.DS[strat:end]
    #         if len(batch_y) == 0: raise StopIteration("Iterative completion")
    #         yield (batch_x, batch_y)
    #         self.index += 1

    def batch(self,im_size=416):
        # inp_dim = random.choice([384, 416, 448, 480, 512, 544, 576, 608, 640, 672])
        start = self.index * self.batch_size
        end = min((self.index + 1) * self.batch_size,self.num_example)
        transforms=get_transform(train=True, multi_scale=False, deaful_size=im_size)
        tmp_img=[]
        tmp_target={"boxes":[],"labels":[]}
        for i in range(start,end):
            img,target=self.DS[i]
            img, target = transforms(img, target)
            tmp_img.append(img)
            tmp_target["boxes"].append(target["boxes"])
            tmp_target["labels"].append(target["labels"])

        self.index += 1

        tmp_img=torch.stack(tmp_img,0)
        tmp_target["boxes"]=torch.stack(tmp_target["boxes"],0)
        tmp_target["labels"]=torch.stack(tmp_target["labels"],0)

        return tmp_img,tmp_target


if __name__=="__main__":
    from datasets import PascalVOCDataset
    import matplotlib.pyplot as plt

    # import sys
    sys.path.append("..")
    from utils.draw import draw_rect
    from utils._util import load_classes

    import pickle

    colors = pickle.load(open("../utils/pallete", "rb"))
    classes = load_classes("../utils/voc.names")

    root = "/media/wucong/d4590a73-a3d9-4971-96fb-4c3cf05abc56/data/VOCdevkit"
    # classes = ["bicycle", "bus", "car", "motorbike", "person"]

    dataset = PascalVOCDataset(root, transforms=None, classes=classes)

    def plot_img(img_list):
        num_imgs = len(img_list)
        rows = np.ceil(num_imgs / 2).astype(np.int)
        cols = np.ceil(num_imgs / rows).astype(np.int)
        _, axis = plt.subplots(rows, cols)

        for ax, img in zip(axis.flatten(), img_list):
            ax.imshow(img)
            ax.axis("off")

        # plt.savefig("preds.jpg")
        plt.show()

    batch_size=4
    for epoch in range(2):
        data_loader = Generator(dataset, batch_size=batch_size, shuffle=True)
        for i in range(data_loader.num_example//batch_size):
            if i%10==0:
                inp_dim = random.choice([384, 416, 448, 480, 512, 544, 576, 608, 640, 672])
            image,target=data_loader.batch(inp_dim)


            img_h, img_w = image.size()[-2:]
            num_images = len(image)
            img_list = []
            labels = target["labels"]
            boxes = target["boxes"] * torch.as_tensor([img_w, img_h, img_w, img_h], dtype=torch.float32,
                                                      device=image.device)[None, :]

            for i in range(num_images):
                tmp_boxes = boxes[i][labels[i] != -1]  # 过滤掉没有的box
                tmp_labels = labels[i][labels[i] != -1]
                tmp_target = {"boxes": tmp_boxes.numpy(), "labels": tmp_labels.numpy()}

                tmp_image = np.asarray(image[i]).transpose([1, 2, 0])
                img = draw_rect(np.clip(tmp_image * 255, 0, 255), tmp_target, classes, colors)
                img_list.append(img)

            plot_img(img_list)

