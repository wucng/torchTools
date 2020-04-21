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

import os
import numpy as np
import torch
from torch import nn
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
try:
    from .tools.engine import train_one_epoch, evaluate
    from .tools import utils
    from .tools import transforms as T
    from ..network import fasterrcnnNet
    from ..loss import yoloLoss
except:
    from tools.engine import train_one_epoch, evaluate
    from tools import utils
    from tools import transforms as T
    import sys
    sys.path.append("..")
    from network import fasterrcnnNet
    from loss import yoloLoss


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
    def __init__(self,trainDP=None,num_classes=2,lr=5e-3,num_epochs = 10,print_freq=10,conf_thres=0.7,nms_thres=0.4,
                 basePath="./models/",save_model="model.pt"):
        super(Fasterrcnn,self).__init__()
        # our dataset has two classes only - background and person
        # num_classes = num_classes
        # let's train it for 10 epochs
        self.num_epochs = num_epochs
        self.print_freq = print_freq
        # seed = 100
        seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}
        # use our dataset and defined transformations
        dataset = PennFudanDataset(trainDP, get_transform(train=True))
        dataset_test = PennFudanDataset(trainDP, get_transform(train=False))

        # split the dataset in train and test set
        # torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True,
            collate_fn=utils.collate_fn,**kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False,
            collate_fn=utils.collate_fn,**kwargs)


        # get the model using our helper function
        # self.network = fasterrcnnNet.FasterRCNN0(num_classes,True)
        self.network = fasterrcnnNet.FasterRCNN1(num_classes,"resnet18",False,use_FPN=False)
        # self.network = fasterrcnnNet.get_instance_segmentation_model(num_classes,True,256) # maskrcnn
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
            lr_scheduler.step()
            # self.eval()
            torch.save(self.network.state_dict(), self.save_model)


        self.test()


    def train(self,epoch):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.network, self.optimizer, self.train_loader, self.device, epoch, self.print_freq)

    def eval(self):
        # evaluate on the test dataset
        evaluate(self.network, self.test_loader, self.device)

    def test(self):
        # pick one image from the test set
        img, _ = self.test_loader[0]
        # put the model in evaluation mode
        self.network.eval()
        with torch.no_grad():
            prediction = self.network([img.to(device)])



if __name__ == "__main__":
    testdataPath = r"C:\practice\data\PennFudanPed\PNGImages"
    traindataPath = r"C:\practice\data\PennFudanPed"

    model = Fasterrcnn(traindataPath,2)

    model()