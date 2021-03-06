try:
    from ..network import yoloNet
    from ..loss import yoloLoss
    from ..datasets import datasets, bboxAug
    from ..visual import opencv
    from ..optm import optimizer
except:
    import sys
    sys.path.append("..")
    from network import yoloNet
    from loss import yoloLoss
    from datasets import datasets, bboxAug
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

class YOLO(nn.Module):
    def __init__(self,network=None,train_dataset=None,test_dataset=None,model_name="resnet18",num_features=None,
                 pretrained=False,dropRate=0.5, usize=256,isTrain=False,mulScale=False,
                 basePath="./",save_model = "model.pt",summaryPath="yolov1_resnet50_416",
                 epochs = 100,print_freq=50,
                 batch_size=2,num_anchors=2,lr=2e-3,
                 threshold_conf=0.5,threshold_cls=0.5, #  # 0.05,0.5
                 conf_thres=0.7,nms_thres=0.4, # 0.8,0.4
                 filter_labels = [],classes=[],version="v1"):# "v1" v2 v3
        super(YOLO,self).__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.print_freq = print_freq
        self.isTrain = isTrain
        self.mulScale = mulScale
        self.classes = classes
        num_classes = len(self.classes)

        # seed = 100
        seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}


        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=collate_fn, **kwargs)
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)

        if version=="v1":
            self.loss_func = yoloLoss.YOLOv1Loss(self.device,num_anchors,num_classes,threshold_conf,
                                         threshold_cls,conf_thres,nms_thres,filter_labels,self.mulScale)
            if network is None:
                self.network = yoloNet.YOLOV1Net(num_classes,num_anchors,model_name,num_features,pretrained,dropRate,usize)
            else:
                self.network = network(num_classes,num_anchors,model_name,num_features,pretrained,dropRate,usize)

        elif version=="v2":
            self.loss_func = yoloLoss.YOLOv2Loss(self.device, num_anchors, num_classes, threshold_conf,
                                                 threshold_cls, conf_thres, nms_thres, filter_labels, self.mulScale)
            if network is None:
                self.network = yoloNet.YOLOV2Net(num_classes, num_anchors, model_name, num_features, pretrained,
                                                 dropRate, usize)
            else:
                self.network = network(num_classes, num_anchors, model_name, num_features, pretrained, dropRate, usize)

        else:
            self.loss_func = yoloLoss.YOLOv3Loss(self.device, num_anchors, num_classes, threshold_conf,
                                                 threshold_cls, conf_thres, nms_thres, filter_labels, self.mulScale)
            if network is None:
                self.network = yoloNet.YOLOV2Net(num_classes, num_anchors, model_name, num_features, pretrained,
                                                 dropRate, usize)
            else:
                self.network = network(num_classes, num_anchors, model_name, num_features, pretrained, dropRate, usize)

        # self.network.apply(yoloNet.weights_init)
        self.network.fpn.apply(yoloNet.weights_init) # backbone 不使用
        self.network.net.apply(yoloNet.weights_init)

        if self.use_cuda:
            self.network.to(self.device)

        if not os.path.exists(basePath):
            os.makedirs(basePath)

        self.save_model = os.path.join(basePath,save_model)
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))
            # self.network.load_state_dict(torch.load(self.save_model,map_location=torch.device('cpu')))

        # optimizer
        base_params = list(
            map(id, self.network.backbone.parameters())
        )
        logits_params = filter(lambda p: id(p) not in base_params, self.network.parameters())

        params = [
            {"params": logits_params, "lr": lr},  # 1e-3
            {"params": self.network.backbone.parameters(), "lr": lr / 10},  # 1e-4
        ]

        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=4e-05)
        # self.optimizer = torch.optim.Adam(params, weight_decay=4e-05)
        self.optimizer = optimizer.RAdam(params, weight_decay=4e-05)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.writer = SummaryWriter(os.path.join(basePath,summaryPath))

        self.history= History()

    def forward(self):
        if self.isTrain:
            for epoch in range(self.epochs):
                loss_record = self.train(epoch)
                # update the learning rate
                self.lr_scheduler.step()
                torch.save(self.network.state_dict(), self.save_model)
                # torch.save(self.network.state_dict(), self.save_model+"_"+str(epoch))

                self.history.epoch.append(epoch)
                for key,value in loss_record.items():
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
        loss_record = {} # 记录每个epoch loss
        for idx, (data, target) in enumerate(self.train_loader):
            if not self.mulScale:
                data = torch.stack(data, 0)  # 不使用多尺度，因此会resize到同一尺度，可以直接按batch计算，加快速度
            if self.use_cuda:
                # data, target = data.to(self.device), target.to(self.device)
                # data = data.to(self.device)
                # target = {k: v.to(self.device) for k, v in target.items()}
                if self.mulScale:
                    data = [d.to(self.device) for d in data]
                else:
                    data = data.to(self.device)
                target = [{k: v.to(self.device) for k, v in targ.items()} for targ in target]

            if self.mulScale:
                output = [self.network(da.unsqueeze(0)) for da in data]
            else:
                output = self.network(data)

            loss_dict = self.loss_func(output, target)

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # 记录到TensorBoard
            self.writer.add_scalar('total_loss', losses.item(), epoch * num_trains // self.batch_size + idx)
            for key, loss in loss_dict.items():
                self.writer.add_scalar(key, loss.item(), epoch * num_trains // self.batch_size + idx)

            if idx % self.print_freq == 0:
                ss = "epoch:{}-({}/{})".format(epoch, idx * self.batch_size, num_trains)
                ss += "\ttotal:{:.3f}".format(losses.item())
                for key, loss in loss_dict.items():
                    ss += "\t{}:{:.3f}".format(key, loss.item())

                print(ss)

            # 记录loss
            for key, loss in loss_dict.items():
                if key not in loss_record:
                    loss_record[key]=0.0
                loss_record[key]+=loss.item()
            if "total" not in loss_record:
                loss_record["total"] = 0.0
            loss_record["total"] += losses.item()

            for key,value in loss_record.items():
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

    def predict(self):
        pass

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