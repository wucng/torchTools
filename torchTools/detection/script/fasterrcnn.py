"""
%%shell

pip install cython
# Install pycocotools, the version by default in Colab
# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

%%shell

# download the Penn-Fudan dataset
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip .
# extract it in the current folder
unzip PennFudanPed.zip

mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
# each mask instance has a different color, from zero to N, where
# N is the number of instances. In order to make visualization easier,
# let's adda color palette to the mask.
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])
mask

%%shell

# Download TorchVision repo to use some files from
# references/detection
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.3.0

cp references/detection/utils.py ../
cp references/detection/transforms.py ../
cp references/detection/coco_eval.py ../
cp references/detection/engine.py ../
cp references/detection/coco_utils.py ../
"""

# !pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# or
# !pip install Cython pycocotools
import os,cv2
import numpy as np
import torch
from torch import nn
import torch.utils.data
from PIL import Image
import torchvision
from tqdm import tqdm
from torchvision import transforms as TT
import matplotlib.pyplot as plt
import time
try:
    from .tools.engine import train_one_epoch, evaluate
    from .tools import utils
    from .tools import transforms as T
    from .tools.nms_pytorch import nms2 as nms
    from ..network import fasterrcnnNet
    from ..visual import opencv,colormap
    # from ..loss import yoloLoss
    from ..datasets.datasets2 import PennFudanDataset,glob_format,PascalVOCDataset
    from ..datasets import bboxAug
except:
    from tools.engine import train_one_epoch, evaluate
    from tools import utils
    from tools import transforms as T
    from tools.nms_pytorch import nms2 as nms
    import sys
    sys.path.append("..")
    from network import fasterrcnnNet
    # from loss import yoloLoss
    from datasets.datasets2 import PennFudanDataset, glob_format,PascalVOCDataset
    from visual import opencv,colormap
    from datasets import bboxAug


def get_transform2(train,advanced=False):
    transforms = []
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms_list = [
            # bboxAug.RandomChoice(),

            # ---------两者取其一--------------------
            bboxAug.RandomHorizontalFlip(),
            bboxAug.RandomTranslate(),
            # bboxAug.RandomRotate(3),
            bboxAug.RandomBrightness(),
            bboxAug.RandomSaturation(),
            bboxAug.RandomHue(),
            bboxAug.RandomBlur(),
            # ---------两者取其一--------------------
            # bboxAug.Augment(advanced),
            # ---------两者取其一--------------------
        ]

        transforms.extend(transforms_list)

    transforms.append(bboxAug.ToTensor())

    return T.Compose(transforms)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


class Fasterrcnn(nn.Module):
    def __init__(self,trainDP=None,classes=[],model_name="resnet101",
                 pretrained=False,usize = 256,use_FPN=False,
                 lr=5e-3,num_epochs = 10,filter_labels=[],
                 print_freq=10,conf_thres=0.7,nms_thres=0.4,
                 batch_size=2,test_batch_size = 2,
                 basePath="./models/",save_model="model.pt",
                 useMask=False,selfmodel=True,typeOfData = "PennFudanDataset"):
        super(Fasterrcnn,self).__init__()
        # our dataset has two classes only - background and person
        # num_classes = num_classes
        # let's train it for 10 epochs
        self.num_epochs = num_epochs
        self.print_freq = print_freq
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.test_batch_size = test_batch_size
        self.classes = classes
        num_classes = len(classes)+1 # 加上背景
        self.filter_labels = filter_labels
        self.useMask = useMask

        seed = 100
        # seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        if typeOfData == "PennFudanDataset":
            Data = PennFudanDataset
        elif typeOfData == "PascalVOCDataset":
            Data = PascalVOCDataset
        else:
            pass
        # use our dataset and defined transformations
        dataset = Data(trainDP, transforms=get_transform(train=True),classes = classes)
        dataset_test = Data(trainDP, transforms=get_transform(train=False),classes = classes)

        # split the dataset in train and test set
        # torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            collate_fn=utils.collate_fn,**kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=batch_size//2 if batch_size//2 else 1, shuffle=False,
            collate_fn=utils.collate_fn,**kwargs)


        if useMask:
            if selfmodel:
                self.network = fasterrcnnNet.get_instance_segmentation_model_cum(num_classes, model_name, pretrained, usize, use_FPN)
            else:
                self.network = fasterrcnnNet.get_instance_segmentation_model(num_classes,pretrained,usize) # maskrcnn
        else:
            if selfmodel:
                # get the model using our helper function
                self.network = fasterrcnnNet.FasterRCNN1(num_classes, model_name, pretrained, usize, use_FPN)
            else:
                self.network = fasterrcnnNet.FasterRCNN0(num_classes,pretrained)

        if self.use_cuda:
            # move model to the right device
            self.network.to(self.device)

        if not os.path.exists(basePath):
            os.makedirs(basePath)

        self.save_model = os.path.join(basePath,save_model)
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))

        # construct an optimizer
        params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, weight_decay=5e-4)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # self.loss_func = yoloLoss.YOLOv1Loss(self.device, conf_thres=conf_thres, nms_thres=nms_thres)

    def forward(self):
        for epoch in range(self.num_epochs):
            self.train(epoch)
            # update the learning rate
            self.lr_scheduler.step()
            # self.eval()
            torch.save(self.network.state_dict(), self.save_model)


    def train(self,epoch):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.network, self.optimizer, self.train_loader, self.device, epoch, self.print_freq)

    def eval(self):
        # evaluate on the test dataset
        evaluate(self.network, self.test_loader, self.device)

    def predict(self,path,nums=None):
        self.network.eval()
        filenames = glob_format(path)
        preprocess = TT.Compose([
            TT.ToTensor()
        ])
        # for i in tqdm(range(len(filenames) // self.test_batch_size)):
        for i in range(len(filenames) // self.test_batch_size):
            if nums is not None and i>nums:break
            temp = filenames[i * self.test_batch_size:(i + 1) * self.test_batch_size]
            input_batch = [preprocess(Image.open(filename).convert("RGB")).to(self.device) for filename in temp]

            with torch.no_grad():
                detections = self.network(input_batch)

            for idx, filename in enumerate(temp):
                # image = cv2.imread(filename)
                image = np.asarray(Image.open(filename).convert("RGB"), np.uint8)
                _detections = self.apply_nms(detections[idx])
                if _detections is None:
                    # cv2.imwrite(filename.replace("image","out"),image)
                    continue
                image = self.draw_rect(image, _detections)
                # cv2.imshow("test", image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                plt.imshow(image)
                plt.show()

                # save
                # newPath = path.replace("PNGImages", "result")
                # if not os.path.exists(os.path.dirname(newPath)): os.makedirs(os.path.dirname(newPath))
                # cv2.imwrite(newPath, image)


    def apply_nms(self,prediction):
        ms = prediction["scores"] > self.conf_thres
        if torch.sum(ms) == 0:
            return None
        else:
            last_scores = []
            last_labels = []
            last_boxes = []
            if self.useMask:last_masks=[]

            scores = prediction["scores"][ms]
            labels = prediction["labels"][ms]
            boxes = prediction["boxes"][ms]
            if self.useMask:masks= prediction["masks"][ms]

            unique_labels = labels.unique()
            for c in unique_labels:
                if c in self.filter_labels:continue

                # Get the detections with the particular class
                temp = labels == c
                _scores = scores[temp]
                _labels = labels[temp]
                _boxes = boxes[temp]
                if self.useMask:_masks = masks[temp]

                if len(_labels) > 1:

                    # keep=py_cpu_nms(_boxes.cpu().numpy(),_scores.cpu().numpy(),self.nms_thres)
                    keep=nms(_boxes,_scores,self.nms_thres)
                    # keep = batched_nms(_boxes, _scores, _labels, self.nms_thres)
                    last_scores.extend(_scores[keep])
                    last_labels.extend(_labels[keep])
                    last_boxes.extend(_boxes[keep])
                    if self.useMask:last_masks.extend(_masks[keep])

                else:
                    last_scores.extend(_scores)
                    last_labels.extend(_labels)
                    last_boxes.extend(_boxes)
                    if self.useMask: last_masks.extend(_masks)

            if self.useMask:
                return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes, "masks": last_masks}

            return {"scores": last_scores, "labels": last_labels, "boxes": last_boxes}

    def draw_rect(self,image, pred,segms=True):
        """segms=False 不画mask"""
        labels = pred["labels"]
        bboxs = pred["boxes"]
        scores = pred["scores"]

        if segms and self.useMask:
            masks = pred["masks"]
            mask_color_id = 0
            color_list = colormap.colormap()

        for idx,(label, bbox, score) in enumerate(zip(labels, bboxs, scores)):
            label = label.cpu().numpy()
            bbox = bbox.cpu().numpy()  # .astype(np.int16)
            score = score.cpu().numpy()
            class_str = "%s:%.3f" % (self.classes[int(label)-1], score)  # 跳过背景
            pos = list(map(int, bbox))
            image = opencv.vis_rect(image, pos, class_str, 0.5, int(label),useMask=False if self.useMask else True)
            if segms and self.useMask:
                mask = masks[idx].cpu().numpy()
                mask = np.clip(np.squeeze(mask, 0) * 255., 0, 255).astype(np.uint8)
                mask_color_id += 1
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                image = opencv.vis_mask(image, mask, color_mask, 0.3, True)

        return image

if __name__ == "__main__":
    classes = ["person"]
    # testdataPath = r"C:\practice\data\PennFudanPed\PNGImages"
    # traindataPath = r"C:\practice\data\PennFudanPed"
    testdataPath = "../../datas/PennFudanPed/PNGImages"
    traindataPath = "../../datas/PennFudanPed"
    basePath = "./models"

    model = Fasterrcnn(traindataPath, classes, "resnet50", pretrained=True, num_epochs=10,
                       conf_thres=0.7, nms_thres=0.4, batch_size=2,usize=256,lr=5e-3,
                       use_FPN=True, basePath=basePath, useMask=False, selfmodel=True)

    model()

    # model.predict(testdataPath,3)

    """
    import torch
    from PIL import Image
    model.network.eval()
    with torch.no_grad():
        for imgs,_ in model.test_loader:
            detections = model.network([img.to("cuda") for img in imgs])
            break
    detections
    
    Image.fromarray(imgs[0].mul(255).permute(1, 2, 0).byte().numpy())
    Image.fromarray(detections[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    """