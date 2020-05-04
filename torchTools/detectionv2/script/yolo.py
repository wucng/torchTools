try:
    from ..tools.engine import train_one_epoch2, evaluate2
    from ..network import net
    from ..loss import yoloLoss
    from ..datasets import datasets, bboxAug
    from ..visual import opencv
    from ..optm import optimizer
    from ..config import config
except:
    import sys
    sys.path.append("..")
    from network import net
    from loss import yoloLoss
    from datasets import datasets, bboxAug
    from visual import opencv
    from optm import optimizer
    from config import config
    from tools.engine import train_one_epoch2, evaluate2

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
import matplotlib; matplotlib.use('TkAgg')
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def collate_fn(batch_data):
    data_list = []
    target_list = []
    for data,target in batch_data:
        data_list.append(data)
        target_list.append(target)

    return data_list,target_list

class YOLO(nn.Module):
    def __init__(self,cfg):
        super(YOLO,self).__init__()

        self.batch_size = cfg["work"]["train"]["batch_size"]
        self.epochs = cfg["work"]["train"]["epochs"]
        self.print_freq = cfg["work"]["train"]["print_freq"]
        self.isTrain = cfg["work"]["train"]["isTrain"]
        self.classes = cfg["work"]["train"]["classes"]
        num_classes = len(self.classes)
        self.train_method = cfg["work"]["train"]["train_method"]
        typeOfData = cfg["work"]["dataset"]["typeOfData"]
        version = cfg["work"]["train"]["version"]


        trainDP = cfg["work"]["dataset"]["trainDataPath"]
        testDP = cfg["work"]["dataset"]["testDataPath"]
        predDP = cfg["work"]["dataset"]["predDataPath"]
        advanced = cfg["work"]["train"]["advanced"]
        useImgaug = cfg["work"]["train"]["useImgaug"]
        resize = cfg["work"]["train"]["resize"]
        use_FPN = cfg["network"]["FPN"]["use_FPN"]
        basePath = cfg["work"]["save"]["basePath"]
        save_model = cfg["work"]["save"]["save_model"]
        summaryPath = cfg["work"]["save"]["summaryPath"]
        # lr = cfg["work"]["train"]["lr"]

        seed = 100
        # seed = int(time.time() * 1000)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.manual_seed(seed)
        kwargs = {'num_workers': 5, 'pin_memory': True} if self.use_cuda else {}

        if useImgaug:
            train_transforms = bboxAug.Compose([
                bboxAug.Augment(advanced=advanced),
                bboxAug.Pad(), bboxAug.Resize(resize, False),
                bboxAug.ToTensor(),  # PIL --> tensor
                bboxAug.Normalize()  # tensor --> tensor
            ])
        else:
            train_transforms = bboxAug.Compose([
                bboxAug.RandomHorizontalFlip(),
                bboxAug.RandomBrightness(),
                bboxAug.RandomBlur(),
                bboxAug.RandomSaturation(),
                bboxAug.RandomHue(),
                bboxAug.RandomRotate(3),
                bboxAug.RandomTranslate(),
                bboxAug.Pad(), bboxAug.Resize(resize, False),
                bboxAug.ToTensor(),  # PIL --> tensor
                bboxAug.Normalize()  # tensor --> tensor
            ])

        test_transforms = bboxAug.Compose([
            bboxAug.Pad(), bboxAug.Resize(resize, False),
            bboxAug.ToTensor(),  # PIL --> tensor
            bboxAug.Normalize()  # tensor --> tensor
        ])

        if self.isTrain:
            if typeOfData=="PennFudanDataset":
                Data = datasets.PennFudanDataset
            elif typeOfData=="PascalVOCDataset":
                Data = datasets.PascalVOCDataset
            else:
                pass
            train_dataset = Data(trainDP,transforms=train_transforms,classes=self.classes)


            if testDP is not None:
                test_dataset = Data(testDP,transforms=test_transforms,classes=self.classes)

            else:
                test_dataset = Data(trainDP,test_transforms, classes=self.classes)
                num_datas = len(train_dataset)
                num_train = int(0.8*num_datas)
                indices = torch.randperm(num_datas).tolist()
                train_dataset = torch.utils.data.Subset(train_dataset, indices[:num_train])
                test_dataset = torch.utils.data.Subset(test_dataset, indices[num_train:])

            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                           collate_fn=collate_fn, **kwargs)

            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)

            if predDP is not None:
                pred_dataset = datasets.ValidDataset(predDP,transforms=test_transforms)

                self.pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False,
                                              collate_fn=collate_fn, **kwargs)
        else:
            pred_dataset = datasets.ValidDataset(predDP,transforms=test_transforms)

            self.pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False,
                                          collate_fn=collate_fn, **kwargs)

        if version=="v1":
            self.loss_func = yoloLoss.YOLOv1Loss(cfg,self.device)
        else:
            self.loss_func = yoloLoss.YOLOv2Loss(cfg, self.device)
        self.network = net.Network(cfg)
        if use_FPN:self.network.fpn.apply(net.weights_init_fpn) # backbone 不使用
        self.network.rpn.apply(net.weights_init_rpn)

        if self.use_cuda:
            self.network.to(self.device)

        if not os.path.exists(basePath):
            os.makedirs(basePath)

        self.save_model = os.path.join(basePath,save_model)
        if os.path.exists(self.save_model):
            self.network.load_state_dict(torch.load(self.save_model))
            # self.network.load_state_dict(torch.load(self.save_model,map_location=torch.device('cpu')))

        self.optimizer = optimizer.build_optimizer(self.network,cfg)
        self.lr_scheduler = optimizer.build_lr_scheduler(self.optimizer,cfg)

        self.writer = SummaryWriter(os.path.join(basePath,summaryPath))

    def forward(self):
        if self.isTrain:
            for epoch in range(self.epochs):
                if self.train_method:
                   self.train(epoch)
                else:
                    self.train2(epoch)
                # update the learning rate
                self.lr_scheduler.step()
                torch.save(self.network.state_dict(), self.save_model)

        else:
            self.test()

    def fit(self):
        pass

    def train2(self,epoch):
            # train for one epoch, printing every 10 iterations
            # train_one_epoch(self.network, self.optimizer, self.train_loader, self.device, epoch, self.print_freq)
            train_one_epoch2(self.network,self.loss_func,self.optimizer,self.train_loader,
                             self.device,epoch,self.print_freq,self.use_cuda)

    def eval(self):
        # evaluate on the test dataset
        evaluate2(self.network,self.loss_func, self.test_loader, self.device)


    def train(self,epoch):
        self.network.train()
        num_trains = len(self.train_loader.dataset)
        for idx, (data, target) in enumerate(self.train_loader):
            data = torch.stack(data, 0)  # 不使用多尺度，因此会resize到同一尺度，可以直接按batch计算，加快速度
            if self.use_cuda:
                data = data.to(self.device)
                target = [{k: v.to(self.device) for k, v in targ.items()} for targ in target]

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

    def predict(self,nums=None):
        self.network.eval()
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.pred_loader):
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
    preddataPath = r"D:\practice\datas\PennFudanPed\PNGImages"
    testdataPath = None
    traindataPath = r"D:\practice\datas\PennFudanPed"
    typeOfData = "PennFudanDataset"
    """
    classes = ["bicycle", "bus", "car", "motorbike", "person"]
    testdataPath = "/home/wucong/practise/datas/valid/PNGImages/"
    traindataPath = "/home/wucong/practise/datas/VOCdevkit/"
    typeOfData = "PascalVOCDataset"
    # """
    basePath = "./models/"
    resize = (224,224)

    cfg = config.get_cfg()
    cfg["work"]["dataset"]["trainDataPath"] = traindataPath
    cfg["work"]["dataset"]["testDataPath"] = testdataPath
    cfg["work"]["dataset"]["predDataPath"] = preddataPath
    cfg["work"]["dataset"]["typeOfData"] = typeOfData
    cfg["work"]["save"]["basePath"] = basePath
    cfg["network"]["backbone"]["model_name"] = "resnet34"
    cfg["network"]["backbone"]["pretrained"] = True
    cfg["work"]["train"]["resize"] = resize
    cfg["work"]["train"]["epochs"] = 50
    cfg["work"]["train"]["classes"] = classes
    cfg["work"]["train"]["useImgaug"] = True
    cfg["work"]["train"]["version"] = "v2"
    cfg["work"]["train"]["method"] = 1
    cfg["network"]["backbone"]["freeze_at"] = "res2"
    cfg["network"]["RPN"]["num_boxes"] = 6 # 2
    cfg["network"]["RPN"]["num_classes"] = len(classes)
    cfg["work"]["loss"]["alpha"] = 0.2
    cfg["work"]["loss"]["threshold_conf"] = 0.2
    cfg["work"]["loss"]["threshold_cls"] = 0.2
    cfg["work"]["loss"]["conf_thres"] = 0.4

    # """
    cfg["network"]["FPN"]["use_FPN"] = True
    cfg["network"]["FPN"]["out_features"] = ["p3","p5"]
    cfg["network"]["RPN"]["in_channels"] = 256
    """
    cfg["network"]["backbone"]["out_features"]=["res5"]
    cfg["network"]["FPN"]["use_FPN"] = False
    cfg["network"]["RPN"]["in_channels"] = 512
    # """
    index = cfg["network"]["backbone"]["index"]
    if cfg["network"]["FPN"]["use_FPN"]:
        name_features = cfg["network"]["FPN"]["name_features"]
        out_features = cfg["network"]["FPN"]["out_features"]
        for out in out_features:
            index.append(name_features.index(out))
    else:
        name_features = cfg["network"]["backbone"]["name_features"]
        out_features = cfg["network"]["backbone"]["out_features"]
        for out in out_features:
            index.append(name_features.index(out))

    cfg["network"]["backbone"]["strides"] =[value for i,value in
                                enumerate(cfg["network"]["backbone"]["strides"]) if i in index]
    cfg["network"]["prioriBox"]["min_sizes"] = [value for i,value in
                                enumerate(cfg["network"]["prioriBox"]["min_sizes"]) if i in index]
    cfg["network"]["prioriBox"]["max_sizes"] = [value for i,value in
                                enumerate(cfg["network"]["prioriBox"]["max_sizes"]) if i in index]
    cfg["network"]["prioriBox"]["aspect_ratios"] = [value for i,value in
                                enumerate(cfg["network"]["prioriBox"]["aspect_ratios"]) if i in index]

    # train_method=1 推荐这种方式训练
    model = YOLO(cfg)

    # model()
    model.predict(5)
    # model.eval()