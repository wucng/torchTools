try:
    from .. import datasets,visual
except:
    import sys
    sys.path.append("..")
    import datasets,visual
import numpy as np
import random
import cv2,os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data,target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list,target_list


def test_datasets():
    # root = r"/media/wucong/225A6D42D4FA828F1/datas/PennFudanPed"
    # classes = ["person"]
    root = "/media/wucong/225A6D42D4FA828F1/datas/voc/VOCdevkit/"
    classes = ["aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
    seed = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    PascalVOCDataset = datasets.datasets.PascalVOCDataset
    # PennFudanDataset = datasets.datasets.PennFudanDataset
    vis_rect = visual.opencv.vis_rect

    bboxAug = datasets.bboxAug
    # dataset = PennFudanDataset(root, transforms=bboxAug.Compose([
    #     bboxAug.ToTensor()
    # ]))

    dataset = PascalVOCDataset(root,  \
           transforms=bboxAug.Compose([
               # bboxAug.RandomChoice(),
               # bboxAug.RandomHorizontalFlip(),
               # bboxAug.RandomBrightness(),
               # bboxAug.RandomBlur(),
               # bboxAug.RandomSaturation(),
               # bboxAug.RandomHue(),
               # bboxAug.RandomRotate(angle=5),
               # bboxAug.RandomTranslate(),
               # bboxAug.Augment(False),
               # bboxAug.Pad(), bboxAug.Resize((416,416), False),
               # bboxAug.ResizeMinMax(800,1333),
               bboxAug.ToTensor(), # PIL --> tensor
               # bboxAug.Normalize() # tensor --> tensor
           ]),classes=classes)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False,collate_fn=collate_fn, **kwargs)

    for datas,targets in data_loader:
        for data,target in zip(datas,targets):
            # from c,h,w ->h,w,c
            data = data.permute(1,2,0)
            # to uint8
            data = torch.clamp(data*255,0,255).to("cpu").numpy().astype(np.uint8)

            # to BGR
            # data = data[...,::-1]
            data = cv2.cvtColor(data,cv2.COLOR_RGB2BGR)

            boxes = target["boxes"].to("cpu").numpy().astype(np.int)
            labels = target["labels"].to("cpu").numpy()
            for box,label in zip(boxes,labels):
                data = vis_rect(data,box,str(label),0.5,label)

            cv2.imshow("test", data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # exit(0)

def test_bboxAug():
    while True:
        im_name = "./2.jpg"
        img = cv2.imread(im_name)
        target = {}
        bboxes = [[73, 35, 326, 467], [333, 188, 560, 469]]
        labels = [0, 1]
        target["boxes"] = torch.as_tensor(bboxes,dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels,dtype=torch.long)

        vis_rect = visual.opencv.vis_rect
        # simple_agu = datasets.bboxAug.simple_agu
        # data, target = simple_agu(img, target, np.random.randint(0, 1000, 1)[0])
        Augment = datasets.bboxAug.Augment
        data, target = Augment(False)(img, target)

        # from c,h,w ->h,w,c
        data = data.permute(1, 2, 0)
        # to uint8
        data = torch.clamp(data * 255, 0, 255).to("cpu").numpy().astype(np.uint8)

        # to BGR
        # data = data[...,::-1]
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        boxes = target["boxes"].to("cpu").numpy().astype(np.int)
        labels = target["labels"].to("cpu").numpy()
        for box, label in zip(boxes, labels):
            data = vis_rect(data, box, str(label), 0.5, label)

        cv2.imshow("test", data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__=="__main__":
    test_datasets()
    # test_bboxAug()