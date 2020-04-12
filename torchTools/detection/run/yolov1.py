try:
    from ..network import net
    from ..loss import loss
    from ..datasets import datasets, bboxAug
    from ..visual import opencv
except:
    import sys
    sys.path.append("..")
    from network import net
    from loss import loss
    from datasets import datasets, bboxAug
    from visual import opencv

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

class YOLOV1(nn.Module):
    def __init__(self,trainDP=None,testDP=None,model_name="resnet18",num_features=None,
                 pretrained=False,droprate=0.5, usize=256,isTrain=False,
                 basePath="./",save_model = "model.pt",summaryPath="yolov1_resnet50_416",
                 epochs = 100,print_freq=1,resize:tuple = (224,224),
                 mulScale=False,advanced=False,batch_size=2,num_anchors=2,lr=2e-4,
                 num_classes=20,
                 threshold_conf=0.5,threshold_cls=0.5, #  # 0.05,0.5
                 conf_thres=0.5,nms_thres=0.4, # 0.8,0.4
                 filter_labels = [],classes=[]):
        super(YOLOV1,self).__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.print_freq = print_freq
        self.isTrain = isTrain
        self.mulScale = mulScale
        self.classes = classes

        # seed = 100
        seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        if self.isTrain:
            train_dataset = datasets.PennFudanDataset(trainDP,
                      transforms=bboxAug.Compose([
                          bboxAug.RandomChoice(),
                          *random.choice([
                              [bboxAug.Pad(), bboxAug.Resize(resize, mulScale)],
                              [bboxAug.Resize2(resize, mulScale)]
                          ]),
                          # ---------两者取其一--------------------
                          bboxAug.RandomHorizontalFlip(),
                          bboxAug.RandomTranslate(),
                          bboxAug.RandomRotate(3),
                          bboxAug.RandomBrightness(),
                          bboxAug.RandomSaturation(),
                          bboxAug.RandomHue(),
                          bboxAug.RandomBlur(),

                          # bboxAug.Augment(advanced),
                          # -------------------------------

                          bboxAug.ToTensor(),  # PIL --> tensor
                          bboxAug.Normalize() # tensor --> tensor
                      ]))

            test_dataset = datasets.ValidDataset(testDP,
                                            transforms=bboxAug.Compose([
                                                bboxAug.Pad(), bboxAug.Resize(resize, False),
                                                bboxAug.ToTensor(),  # PIL --> tensor
                                                bboxAug.Normalize()  # tensor --> tensor
                                            ]))

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=collate_fn, **kwargs)

            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)


        else:
            test_dataset = datasets.ValidDataset(testDP,
                      transforms=bboxAug.Compose([
                          bboxAug.Pad(), bboxAug.Resize(resize, False),
                          bboxAug.ToTensor(),  # PIL --> tensor
                          bboxAug.Normalize() # tensor --> tensor
                      ]))

            self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)



        self.loss_func = loss.YOLOv1Loss(self.device,num_anchors,num_classes,threshold_conf,
                                         threshold_cls,conf_thres,nms_thres,filter_labels,self.mulScale)
        self.network = net.YOLOV1Net(num_classes,num_anchors,model_name,num_features,pretrained,droprate,usize)
        self.network.apply(net.weights_init)
        if self.use_cuda:
            self.network.to(self.device)

        if not os.path.exists(basePath):
            os.makedirs(basePath)

        self.save_model = os.path.join(basePath,save_model)
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))

        # optimizer
        base_params = list(
            map(id, self.network.backbone.parameters())
        )
        logits_params = filter(lambda p: id(p) not in base_params, self.network.parameters())

        params = [
            {"params": logits_params, "lr": lr},  # 1e-3
            {"params": self.network.backbone.parameters(), "lr": lr / 5},  # 1e-4
        ]

        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=4e-05)
        self.optimizer = torch.optim.Adam(params, weight_decay=4e-05)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        self.writer = SummaryWriter(os.path.join(basePath,summaryPath))

    def forward(self):
        if self.isTrain:
            for epoch in range(self.epochs):
                self.train(epoch)
                self.test()
                # update the learning rate
                self.lr_scheduler.step()
                torch.save(self.network.state_dict(), self.save_model)
                # torch.save(self.network.state_dict(), self.save_model+"_"+str(epoch))
        else:
            self.test()

    def fit(self):
        pass

    def train(self,epoch):
        self.network.train()
        num_trains = len(self.train_loader.dataset)
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

    def test(self):
        self.network.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                if idx >2:break # ???????????????????????
                data = torch.stack(data,0) # 做测试时不使用多尺度，因此会resize到同一尺度，可以直接按batch计算，加快速度
                if self.use_cuda:
                    data = data.to(self.device)
                    # data = [d.to(self.device) for d in data]
                    new_target = [{k: v.to(self.device) for k, v in targ.items() if k!="path"} for targ in target]

                output = self.network(data)
                preds = self.loss_func(output,new_target)

                for i in range(len(target)):
                    pred = preds[i]
                    if pred is None:continue
                    path = target[i]["path"]
                    image = np.asarray(PIL.Image.open(path).convert("RGB"), np.uint8)
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    image = draw_rect(image,pred,self.classes)

                    # cv2.imshow("test", image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    plt.imshow(image)
                    plt.show()
                    # PIL.Image.fromarray(image).show()


    def predict(self):
        pass

def draw_rect(image,pred,classes):
    labels = pred["labels"]
    bboxs = pred["boxes"]
    scores = pred["scores"]

    for label,bbox,score in zip(labels,bboxs,scores):
        label=label.cpu().numpy()
        bbox=bbox.cpu().numpy()#.astype(np.int16)
        score=score.cpu().numpy()
        class_str="%s:%.3f"%(classes[int(label)],score) # 跳过背景
        pos = list(map(int, bbox))

        image=opencv.vis_rect(image,pos,class_str,0.5,int(label))
    return image


if __name__=="__main__":
    classes=["__background__","person"]
    # testdataPath = "/kaggle/input/PennFudanPed/PNGImages/"
    testdataPath = "C:/Users/MI/Documents/GitHub/PennFudanPed/PNGImages/"
    # traindataPath = "/kaggle/input/"
    traindataPath = "C:/Users/MI/Documents/GitHub/PennFudanPed/"
    model = YOLOV1(traindataPath,testdataPath,isTrain=True,num_anchors=2,num_classes=1,mulScale=True)
    model()