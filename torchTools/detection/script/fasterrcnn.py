"""
classes = ["bicycle", "bus", "car","motorbike", "person"]
dataset = PascalVOCDataset(root, transforms=get_transform(train=True), classes=classes)
dataset_test = PascalVOCDataset(root, transforms=get_transform(train=False), classes=classes)

# our dataset has two classes only - background and person
num_classes = len(classes) + 1

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50]) # -50
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:]) # -50
"""

try:
    from ..network import fasterrcnnNet
    from ..loss import yoloLoss
    from ..datasets import datasets2, bboxAug
    from ..visual import opencv
    from ..optm import optimizer
    from .tools import utils
    # from .tools.engine import *
except:
    from tools import utils
    # from tools.engine import *
    import sys
    sys.path.append("..")
    from network import fasterrcnnNet
    from loss import yoloLoss
    from datasets import datasets2, bboxAug
    from visual import opencv
    from optm import optimizer

from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
import cv2,os,time
from PIL import Image
import PIL.Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import math

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data,target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list,target_list

class History():
    epoch = []
    history = {}

    # 打印训练结果信息
    def show_final_history(self):
        fig, ax = plt.subplots(1, len(self.history), figsize=(15, 5))
        for i,(k,v) in enumerate(self.history.items()):
            ax[i].set_title(k)
            ax[i].plot(self.epoch,v,label=k)
            ax[i].legend()

        plt.show()

    def show(self):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.set_title("train loss")
        for i,(k,v) in enumerate(self.history.items()):
            # ax.set_title(k)
            ax.plot(self.epoch,v,label=k)
        ax.legend()
        plt.show()

class FasterRCNN(nn.Module):
    def __init__(self,trainDP=None,testDP=None,model_name="resnet18",num_features=None,
                 pretrained=False,dropRate=0.5,hideSize=64, usize=256,use_FPN=False,isTrain=False,
                 basePath="./",save_model = "model.pt",summaryPath="yolov1_resnet50_416",
                 epochs = 100,print_freq=1,resize:tuple = (224,224),
                 mulScale=False,advanced=False,batch_size=2,num_anchors=2,lr=2e-3,
                 typeOfData="PennFudanDataset",
                 threshold_conf=0.5,threshold_cls=0.5, #  # 0.05,0.5
                 conf_thres=0.5,nms_thres=0.4, # 0.8,0.4
                 filter_labels = [],classes=[]):# "v1" v2 v3
        super(FasterRCNN,self).__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.print_freq = print_freq
        self.isTrain = isTrain
        self.mulScale = mulScale
        self.classes = classes
        num_classes = len(self.classes)+1 # 包括背景

        # seed = 100
        seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        if self.isTrain:
            if typeOfData=="PennFudanDataset":
                Data = datasets2.PennFudanDataset
            elif typeOfData=="PascalVOCDataset":
                Data = datasets2.PascalVOCDataset
            else:
                pass
            train_dataset = Data(trainDP,
                      transforms=bboxAug.Compose([
                          # bboxAug.RandomChoice(),
                          bboxAug.Pad(), bboxAug.Resize(resize, mulScale),
                          # *random.choice([
                          #     [bboxAug.Pad(), bboxAug.Resize(resize, mulScale)],
                          #     [bboxAug.Resize2(resize, mulScale)]
                          # ]),

                          # ---------两者取其一--------------------
                          # bboxAug.RandomHorizontalFlip(),
                          # bboxAug.RandomTranslate(),
                          # # bboxAug.RandomRotate(3),
                          # bboxAug.RandomBrightness(),
                          # bboxAug.RandomSaturation(),
                          # bboxAug.RandomHue(),
                          # bboxAug.RandomBlur(),

                          # bboxAug.Augment(advanced),
                          # -------------------------------

                          bboxAug.ToTensor(),  # PIL --> tensor
                          # bboxAug.Normalize() # tensor --> tensor
                      ]),classes=classes)

            test_dataset = datasets2.ValidDataset(testDP,
                                            transforms=bboxAug.Compose([
                                                bboxAug.Pad(), bboxAug.Resize(resize, False),
                                                bboxAug.ToTensor(),  # PIL --> tensor
                                                # bboxAug.Normalize()  # tensor --> tensor
                                            ]))

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=collate_fn, **kwargs)

            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)

        else:
            test_dataset = datasets2.ValidDataset(testDP,
                      transforms=bboxAug.Compose([
                          bboxAug.Pad(), bboxAug.Resize(resize, False),
                          bboxAug.ToTensor(),  # PIL --> tensor
                          # bboxAug.Normalize() # tensor --> tensor
                      ]))

            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)


        self.network = fasterrcnnNet.FasterRCNN1(num_classes,model_name,pretrained,hideSize,usize,use_FPN)
        # self.network = fasterrcnnNet.FasterRCNN0(num_classes,pretrained)
        self.loss_func = yoloLoss.YOLOv1Loss(self.device, num_anchors, num_classes, threshold_conf,
                                             threshold_cls, conf_thres, nms_thres, filter_labels, self.mulScale)

        if self.use_cuda:
            self.network.to(self.device)

        if not os.path.exists(basePath):
            os.makedirs(basePath)

        self.save_model = os.path.join(basePath,save_model)
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))
            # self.network.load_state_dict(torch.load(self.save_model,map_location=torch.device('cpu')))

        # construct an optimizer
        params = [p for p in self.network.parameters() if p.requires_grad]

        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=4e-05)
        # self.optimizer = torch.optim.Adam(params, weight_decay=4e-05)
        self.optimizer = optimizer.RAdam(params, weight_decay=4e-05)
        # optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.writer = SummaryWriter(os.path.join(basePath,summaryPath))

        self.history = History()

    def forward(self):
        if self.isTrain:
            for epoch in range(self.epochs):
                loss_record = self.train(epoch)
                # if epoch>0 and epoch%30==0:
                #     self.test(3)
                # update the learning rate
                self.lr_scheduler.step()
                torch.save(self.network.state_dict(), self.save_model)
                # torch.save(self.network.state_dict(), self.save_model+"_"+str(epoch))

                # evaluate on the test dataset
                # evaluate(self.network, data_loader_test, device=device)

                self.history.epoch.append(epoch)
                for key, value in loss_record.items():
                    if key not in self.history.history:
                        self.history.history[key] = []
                    self.history.history[key].append(value)

        else:
            self.test()

    def fit(self):
        pass

    def train(self,epoch):
        self.network.train()
        num_trains = len(self.train_loader.dataset)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, num_trains - 1)

            lr_scheduler = utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        loss_record = {}  # 记录每个epoch loss
        # for idx, (data, target) in enumerate(self.train_loader):
        for idx, (images, targets) in enumerate(metric_logger.log_every(self.train_loader, self.print_freq, header)):
            if self.use_cuda:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.network(images, targets)
            # RuntimeError: No such operator torchvision::nms
            # 手动修改 torchvision.ops.boxes  nms 修改下

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # 记录到TensorBoard
            self.writer.add_scalar('total_loss', losses.item(), epoch * num_trains // self.batch_size + idx)
            for key, loss in loss_dict.items():
                self.writer.add_scalar(key, loss.item(), epoch * num_trains // self.batch_size + idx)

            # if idx % self.print_freq == 0:
            #     ss = "epoch:{}-({}/{})".format(epoch, idx * self.batch_size, num_trains)
            #     ss += "\ttotal:{:.3f}".format(losses.item())
            #     for key, loss in loss_dict.items():
            #         ss += "\t{}:{:.3f}".format(key, loss.item())
            #
            #     print(ss)

            # 记录loss
            for key, loss in loss_dict.items():
                if key not in loss_record:
                    loss_record[key] = 0.0
                loss_record[key] += loss.item()
            if "total" not in loss_record:
                loss_record["total"] = 0.0
            loss_record["total"] += losses.item()

            for key, value in loss_record.items():
                loss_record[key] /= num_trains

        return loss_record


    def test(self,nums=None):
        self.network.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                if nums is not None:
                    if idx > nums:break
                data = torch.stack(data,0) # 做测试时不使用多尺度，因此会resize到同一尺度，可以直接按batch计算，加快速度
                if self.use_cuda:
                    data = data.to(self.device)
                    # data = [d.to(self.device) for d in data]
                    new_target = [{k: v.to(self.device) for k, v in targ.items() if k!="path"} for targ in target]
                else:
                    new_target = target

                output = self.network(data)
                # preds = self.loss_func(output,new_target)
                preds = self.loss_func.apply_nms(output)

                for i in range(len(target)):
                    pred = preds[i]
                    if pred is None:continue
                    path = target[i]["path"]
                    image = np.asarray(PIL.Image.open(path).convert("RGB"), np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = self.draw_rect(image,pred)

                    cv2.imshow("test", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    # plt.imshow(image)
                    # plt.show()
                    # PIL.Image.fromarray(image).show()

                    # save
                    # newPath = path.replace("PNGImages", "result")
                    # if not os.path.exists(os.path.dirname(newPath)): os.makedirs(os.path.dirname(newPath))
                    # cv2.imwrite(newPath, image)

    def predict(self,nums=None):
        self.network.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                if nums is not None:
                    if idx > nums:break
                data = torch.stack(data,0) # 做测试时不使用多尺度，因此会resize到同一尺度，可以直接按batch计算，加快速度
                if self.use_cuda:
                    data = data.to(self.device)
                    # data = [d.to(self.device) for d in data]
                    new_target = [{k: v.to(self.device) for k, v in targ.items() if k!="path"} for targ in target]
                else:
                    new_target = target

                output = self.network(data)
                preds = self.loss_func(output,new_target)

                for i in range(len(target)):
                    pred = preds[i]
                    if pred is None:continue
                    path = target[i]["path"]
                    image = np.asarray(PIL.Image.open(path).convert("RGB"), np.uint8)
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = self.draw_rect(image,pred)

                    # cv2.imshow("test", image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    plt.imshow(image)
                    plt.show()
                    # PIL.Image.fromarray(image).show()

                    # save
                    # newPath = path.replace("PNGImages", "result")
                    # if not os.path.exists(os.path.dirname(newPath)): os.makedirs(os.path.dirname(newPath))
                    # cv2.imwrite(newPath, image)


    def draw_rect(self,image,pred):
        labels = pred["labels"]
        bboxs = pred["boxes"]
        scores = pred["scores"]

        for label,bbox,score in zip(labels,bboxs,scores):
            label=label.cpu().numpy()
            bbox=bbox.cpu().numpy()#.astype(np.int16)
            score=score.cpu().numpy()
            class_str="%s:%.3f"%(self.classes[int(label)],score) # 跳过背景
            pos = list(map(int, bbox))

            image=opencv.vis_rect(image,pos,class_str,0.5,int(label))
        return image


if __name__=="__main__":
    # """
    classes = ["person"]
    # testdataPath = "/home/wucong/practise/datas/valid/PNGImages/"
    # traindataPath = "/home/wucong/practise/datas/PennFudanPed/"
    # testdataPath = "D:/practice/datas/PennFudanPed/PNGImages/"
    # traindataPath = "D:/practice/datas/PennFudanPed/"
    testdataPath = r"C:\practice\data\PennFudanPed\PNGImages"
    traindataPath = r"C:\practice\data\PennFudanPed"
    typeOfData = "PennFudanDataset"
    """
    classes = ["bicycle", "bus", "car", "motorbike", "person"]
    testdataPath = "/home/wucong/practise/datas/valid/PNGImages/"
    traindataPath = "/home/wucong/practise/datas/VOCdevkit/"
    typeOfData = "PascalVOCDataset"
    # """

    basePath = "./models/"
    model = FasterRCNN(traindataPath, testdataPath, "resnet18", pretrained=False, num_features=1,resize=(112,112),
                   isTrain=True, num_anchors=3, mulScale=False, epochs=400, print_freq=40,dropRate=0.5,
                   basePath=basePath, threshold_conf=0.5, threshold_cls=0.5, lr=2e-3, batch_size=2,
                   conf_thres=0.7, nms_thres=0.4, classes=classes,typeOfData=typeOfData,usize=256)

    model()
