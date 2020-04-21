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
    root = r"C:\practice\data\PennFudanPed"
    seed = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(seed)
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    # PennFudanDataset = datasets.datasets.PennFudanDataset
    PennFudanDataset = datasets.datasets2.PennFudanDataset
    vis_rect = visual.opencv.vis_rect

    # dataset = PennFudanDataset(root, transforms=None)

    bboxAug = datasets.bboxAug
    dataset = PennFudanDataset(root,  \
           transforms=bboxAug.Compose([
               # random.choice([
               #     bboxAug.RandomCrop(),
               #     bboxAug.RandomScale2(),
               #     bboxAug.RandomScale(),
               #     bboxAug.RandomDrop((0.05, 0.05)),
               # ]),
               # bboxAug.RandomChoice(),
               *random.choice([
                   # [bboxAug.Pad(),bboxAug.Resize((256,256),True)],
                   [bboxAug.Resize2((256,256),True)]
               ]),
               # bboxAug.Augment(True),
               bboxAug.ToTensor(), # PIL --> tensor
               # bboxAug.Normalize() # tensor --> tensor
           ]))

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